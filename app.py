import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
from keras.models import load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from joblib import load
from PIL import Image
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import os
import io
import gc
from google.oauth2 import service_account
from concurrent.futures import ThreadPoolExecutor


# --- Configuración de Streamlit ---
st.set_page_config(
    page_title="Embryo Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Embryo Transfer Prioritization with AI")
st.sidebar.markdown("### Configuración del Modelo")
threshold = st.sidebar.slider("Umbral de Transferibilidad", 0.5, 1.0, 0.8)

# --- Configuración de Google Drive ---
@st.cache_resource
def get_drive_service():
    creds_info = {
        "type": os.getenv('SERVICE_ACCOUNT_TYPE'),
        "project_id": os.getenv('project_id'),
        "private_key_id": os.getenv('private_key_id'),
        "private_key": os.getenv('private_key').replace('\\n', '\n'),
        "client_email": os.getenv('client_email'),
        "client_id": os.getenv('client_id'),
        "auth_uri": os.getenv('auth_uri'),
        "token_uri": os.getenv('token_uri'),
        "auth_provider_x509_cert_url": os.getenv('auth_provider_x509_cert_url'),
        "client_x509_cert_url": os.getenv('client_x509_cert_url'),
    }
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=['https://www.googleapis.com/auth/drive'])
    return build('drive', 'v3', credentials=creds)

drive_service = get_drive_service()

def download_from_drive(file_id, destination_path):
    request = drive_service.files().get_media(fileId=file_id)
    with io.BytesIO() as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        with open(destination_path, "wb") as f:
            f.write(fh.read())

# --- Cargar modelos ---
@st.cache_resource
def load_models():
    keras_model_path = "modelo_T_D_final.keras"
    rf_model_path = "modelo_randomforest_blanco.pkl"

    # Descargar modelos desde Google Drive
    download_from_drive(os.getenv("MODEL_KERAS_ID"), keras_model_path)
    download_from_drive(os.getenv("RF_MODEL_ID"), rf_model_path)

    # Cargar modelos
    transfer_model = load_model(keras_model_path)
    rf_model = load(rf_model_path)

    # Modelo VGG16 recortado
    vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = vgg16_base.get_layer('block2_pool').output
    x = GlobalAveragePooling2D()(x)
    vgg_model = Model(inputs=vgg16_base.input, outputs=x)

    return transfer_model, rf_model, vgg_model

transfer_model, rf_model, vgg_model = load_models()

# --- Preprocesamiento de fotogramas ---
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    return preprocess_input(frame_resized)

# --- Procesamiento del video en *streaming* ---
def process_batch(frames_batch, vgg_model, rf_model):
    features = vgg_model.predict(np.array(frames_batch), batch_size=len(frames_batch))
    rf_predictions = rf_model.predict(features)
    results = [(i, pred) for i, pred in enumerate(rf_predictions) if pred == 1]
    return results

def process_video_vgg_rf_parallel(video_path, vgg_model, rf_model, batch_size=32):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_results = []
    frames_batch = []
    frame_numbers_batch = []

    st.write("Procesando video en paralelo...")
    progress_bar = st.progress(0)
    processed_frames = 0

    with ThreadPoolExecutor() as executor:
        futures = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frames_batch.append(preprocess_frame(frame))
            frame_numbers_batch.append(frame_number)

            if len(frames_batch) == batch_size:
                # Enviar el batch a un thread
                futures.append(executor.submit(process_batch, frames_batch, vgg_model, rf_model))

                # Resetear el batch
                frames_batch = []
                frame_numbers_batch = []

            processed_frames += 1
            progress_bar.progress(processed_frames / total_frames)

        # Procesar frames restantes
        if frames_batch:
            futures.append(executor.submit(process_batch, frames_batch, vgg_model, rf_model))

        # Recoger resultados
        for future in futures:
            frame_results.extend(future.result())

    cap.release()
    progress_bar.empty()
    gc.collect()  # Liberar memoria
    return frame_results

# --- Predicción con el modelo Keras ---
def process_video_keras(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keras_results = []

    st.write("Procesando video con modelo Keras...")
    progress_bar = st.progress(0)
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_preprocessed = preprocess_frame(frame)
        prediction = transfer_model.predict(np.expand_dims(frame_preprocessed, axis=0))[0][0]
        keras_results.append((frame_number, prediction))

        processed_frames += 1
        progress_bar.progress(processed_frames / total_frames)

    cap.release()
    progress_bar.empty()
    return keras_results

# --- Generación del gráfico ---
def generate_plot(frame_results_rf, keras_results, threshold):
    valid_frames = {f for f, pred in frame_results_rf if pred == 1}
    filtered_results = [(f, p) for f, p in keras_results if f in valid_frames]

    if not filtered_results:
        st.warning("No hay frames transferibles según los modelos.")
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[f for f, _ in filtered_results],
        y=[p for _, p in filtered_results],
        mode='markers',
        marker=dict(size=8, color=[
            'rgb(33, 164, 174)' if p > threshold else 'rgb(230, 100, 85)'
            for _, p in filtered_results
        ]),
        text=[f"Prob: {p:.2f}" for _, p in filtered_results]
    ))
    fig.update_layout(
        xaxis_title="Frame",
        yaxis_title="Probabilidad de Transferibilidad",
        yaxis=dict(range=[0, 1])
    )
    return fig

# --- Interfaz de usuario ---
video_file = st.file_uploader("Sube un video", type=['mp4', 'avi', 'mov'])
if video_file:
    temp_video_path = f"/tmp/{video_file.name}"
    with open(temp_video_path, 'wb') as f:
        f.write(video_file.read())

    st.video(temp_video_path)

    if st.button("Procesar Video"):
        st.write("Procesando video con VGG16 + RF...")
        frame_results_rf = process_video_vgg_rf_parallel(temp_video_path, vgg_model, rf_model)
        st.write("Procesando video con modelo Keras...")
        keras_results = process_video_keras(temp_video_path)
        st.write("Generando gráfico...")
        fig = generate_plot(frame_results_rf, keras_results, threshold)
        if fig:
            st.plotly_chart(fig)
    
        # Liberar memoria después de procesar
        del video_file, temp_video_path
        gc.collect()


