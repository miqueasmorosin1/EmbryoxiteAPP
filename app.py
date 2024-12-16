import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
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
from google.oauth2 import service_account
import gc
from keras import backend as K

st.set_page_config(
    page_title="Embryo Analysis",
    page_icon="apl/image.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Fondo dinámico usando HTML y CSS ---
def set_image_background():
    image_url = "https://raw.githubusercontent.com/tu-usuario/tu-repositorio/main/apl/image1.png"  # URL directa a tu imagen

    css = f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Llamada a la función
set_image_background()

# --- Contenido de la aplicación ---
st.title("Embryo Transfer Prioritization")
threshold = 0.8

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
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    with open(destination_path, "wb") as f:
        f.write(fh.getvalue())

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
    frame_preprocessed = preprocess_input(frame_resized)
    return frame_preprocessed

# --- Comparar últimos 10 frames con una imagen ---
def compare_last_frames_with_image(video_path, image_path, similarity_threshold=0.9):
    """
    Compara los últimos 10 frames de un video con una imagen.
    Elimina frames similares a la imagen basada en un umbral de similitud.
    """
    # Leer la imagen de referencia
    reference_image = cv2.imread(image_path)
    if reference_image is None:
        st.error(f"No se pudo cargar la imagen de referencia desde {image_path}")
        return set()

    reference_image = cv2.resize(reference_image, (224, 224))
    reference_hist = cv2.calcHist([reference_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    reference_hist = cv2.normalize(reference_hist, reference_hist).flatten()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_frames = max(0, total_frames - 10)

    similar_frames = set()

    cap.set(cv2.CAP_PROP_POS_FRAMES, last_frames)  # Ir a los últimos 10 frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_resized = cv2.resize(frame, (224, 224))
        frame_hist = cv2.calcHist([frame_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        frame_hist = cv2.normalize(frame_hist, frame_hist).flatten()

        # Comparar similitud entre histogramas
        similarity = cv2.compareHist(reference_hist, frame_hist, cv2.HISTCMP_CORREL)
        if similarity >= similarity_threshold:
            similar_frames.add(frame_number)

    cap.release()
    return similar_frames

# --- Procesamiento del video con VGG16-RF en batches paralelos ---
def process_video_vgg_rf_batches(video_path, batch_size=32):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_results = []
    frames_batch = []
    frame_numbers_batch = []
    progress_bar = st.progress(0)
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frames_batch.append(preprocess_frame(frame))
        frame_numbers_batch.append(frame_number)

        if len(frames_batch) == batch_size:
            features = vgg_model.predict(np.array(frames_batch), batch_size=batch_size)
            rf_predictions = rf_model.predict(features)

            frame_results.extend([
                (frame_numbers_batch[i], rf_predictions[i])
                for i in range(len(rf_predictions)) if rf_predictions[i] == 1
            ])

            processed_frames += len(frames_batch)
            progress_bar.progress(processed_frames / total_frames)

            frames_batch = []
            frame_numbers_batch = []

    if frames_batch:
        features = vgg_model.predict(np.array(frames_batch), batch_size=len(frames_batch))
        rf_predictions = rf_model.predict(features)
        frame_results.extend([
            (frame_numbers_batch[i], rf_predictions[i])
            for i in range(len(rf_predictions)) if rf_predictions[i] == 1
        ])

    cap.release()
    progress_bar.empty()
    return frame_results

# --- Predicción con el modelo Keras frame por frame ---
def process_all_frames_with_keras(video_path, batch_size=2):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keras_results = []

    frames_batch = []
    frame_numbers_batch = []
    progress_bar = st.progress(0)
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frames_batch.append(preprocess_frame(frame))
        frame_numbers_batch.append(frame_number)

        if len(frames_batch) == batch_size:
            predictions = transfer_model.predict(np.array(frames_batch), batch_size=batch_size)
            keras_results.extend(zip(frame_numbers_batch, predictions[:, 0]))

            processed_frames += len(frames_batch)
            progress_bar.progress(processed_frames / total_frames)

            frames_batch = []
            frame_numbers_batch = []

    if frames_batch:
        predictions = transfer_model.predict(np.array(frames_batch), batch_size=len(frames_batch))
        keras_results.extend(zip(frame_numbers_batch, predictions[:, 0]))

    cap.release()
    progress_bar.empty()
    return keras_results

# --- Generación del gráfico ---
def generate_plot_vgg_keras(frame_results_rf, keras_results, threshold=0.8):
    valid_frame_numbers = {frame_number for frame_number, pred in frame_results_rf if pred == 1}
    filtered_keras_results = [
        (frame_number, prob) for frame_number, prob in keras_results if frame_number in valid_frame_numbers
    ]

    if not filtered_keras_results:
        st.warning("No hay frames transferibles según los modelos.")
        return None

    frame_numbers = [frame_number for frame_number, _ in filtered_keras_results]
    transfer_probs = [prob for _, prob in filtered_keras_results]

    colors = [
        'rgb(33, 164, 174)' if prob > threshold else 'rgb(230, 100, 85)'
        for prob in transfer_probs
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame_numbers,
        y=transfer_probs,
        mode='markers',
        marker=dict(color=colors, size=8),
        text=[f'Probabilidad: {p:.2f}' for p in transfer_probs]
    ))
    fig.update_layout(
        xaxis_title='Frame',
        yaxis_title='Probabilidad de Transferibilidad',
        title='Predicciones por Frame - Modelo Keras',
        yaxis=dict(
            tickvals=[0, 1],
            ticktext=['En Desarrollo', 'Transferido'],
            range=[0, 1]
        )
    )
    return fig

def cleanup():
    K.clear_session()  # Limpia memoria usada por Keras/TensorFlow
    gc.collect() 
# --- Interfaz de usuario ---
col1, col2 = st.columns(2)
with col1:
    video_file = st.file_uploader("Sube un video", type=['mp4', 'avi', 'mov'])
    if video_file:
        temp_video_path = f"/tmp/{video_file.name}"
        with open(temp_video_path, 'wb') as f:
            f.write(video_file.read())
    
        st.video(temp_video_path, start_time=0)
    
        if st.button("Procesar Video"):
            progress_message = st.empty()

            # Limpia recursos previos
            gc.collect()
            
            # Proceso 1: Eliminando errores
            progress_message.write("Eliminando Errores...")
            frame_results_rf = process_video_vgg_rf_batches(temp_video_path)
            progress_message.empty()
            
            # Proceso 2: Procesando video
            progress_message.write("Procesando Video...")
            keras_results = process_all_frames_with_keras(temp_video_path, batch_size=2)
            progress_message.empty()
            
            # Proceso 3: Comparando últimos frames
            progress_message.write("Comparando últimos frames con la imagen de referencia...")
            image_path = "apl/apl_Missing.png"
            similar_frames = compare_last_frames_with_image(temp_video_path, image_path)
            progress_message.empty()
    
            frame_results_rf = [(frame_number, pred) for frame_number, pred in frame_results_rf if frame_number not in similar_frames]
            keras_results = [(frame_number, prob) for frame_number, prob in keras_results if frame_number not in similar_frames]
            results = (frame_results_rf, keras_results)

            # Liberar recursos después del procesamiento
            del video_file
            cleanup()
        else:
            results = None
    else:
        results = None
with col2:
    if results:
        frame_results_rf, keras_results = results
        st.write("Generando gráfico...")
        fig = generate_plot_vgg_keras(frame_results_rf, keras_results, threshold=threshold)
        if fig:
            st.plotly_chart(fig)
    else:
        st.info("Cargue un video y presione 'Procesar Video' para ver el gráfico.")
