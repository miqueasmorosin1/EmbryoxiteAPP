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
from skimage.metrics import structural_similarity as ssim

# --- Configuración de Streamlit ---
st.set_page_config(
    page_title="Embryo Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
#hola
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

@st.cache_resource
def load_reference_image():
    ref_image_path = "apl/apl_Missing.png"
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    ref_image_resized = cv2.resize(ref_image, (224, 224))  # Ajustar al tamaño esperado
    return ref_image_resized

reference_image = load_reference_image()
# --- Comparar frames con la imagen de referencia ---
def is_similar_to_reference(frame, reference, threshold=0.8):
    """
    Compara un frame con la imagen de referencia usando SSIM.
    Devuelve True si la similitud es mayor al umbral, False en caso contrario.
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    frame_resized = cv2.resize(frame_gray, (224, 224))  # Ajustar al tamaño esperado

    similarity, _ = ssim(frame_resized, reference, full=True)
    return similarity > threshold

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

            # Filtro estricto: Guardar solo frames etiquetados como "Embrion" (1)
            frame_results.extend([
                (frame_numbers_batch[i], rf_predictions[i])
                for i in range(len(rf_predictions)) if rf_predictions[i] == 1
            ])

            processed_frames += len(frames_batch)
            progress_bar.progress(processed_frames / total_frames)

            # Resetear los batches
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

    # Procesar frames restantes
    if frames_batch:
        predictions = transfer_model.predict(np.array(frames_batch), batch_size=len(frames_batch))
        keras_results.extend(zip(frame_numbers_batch, predictions[:, 0]))

    cap.release()
    progress_bar.empty()
    return keras_results

# --- Generación del gráfico ---
def generate_plot_vgg_keras(frame_results_rf, keras_results, threshold=0.8):
    # Filtrar frames según las predicciones del modelo VGG16-RF
    valid_frame_numbers = {frame_number for frame_number, pred in frame_results_rf if pred == 1}
    filtered_keras_results = [
        (frame_number, prob) for frame_number, prob in keras_results if frame_number in valid_frame_numbers
    ]

    # Verificar similitud de los últimos 10 frames con la imagen de referencia
    if filtered_keras_results:
        last_10_frames = filtered_keras_results[-10:]
        filtered_keras_results = []
        for frame_number, prob in last_10_frames:
            cap = cv2.VideoCapture(video_file.name)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = cap.read()
            cap.release()

            if not ret or is_similar_to_reference(frame, reference_image):
                continue  # Omitir si es similar a la referencia

            filtered_keras_results.append((frame_number, prob))

    # Generar el gráfico
    if not filtered_keras_results:
        st.warning("No hay frames transferibles según los modelos o tras la comparación con la imagen de referencia.")
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
        yaxis=dict(range=[0, 1])  # Rango de probabilidades entre 0 y 1
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
        st.write("Eliminando Errores...")
        frame_results_rf = process_video_vgg_rf_batches(temp_video_path)
    
        st.write("Procesando Video...")
        keras_results = process_all_frames_with_keras(temp_video_path, batch_size=2)
    
        st.write("Generando gráfico...")
        fig = generate_plot_vgg_keras(frame_results_rf, keras_results, threshold=threshold)
        if fig:
            st.plotly_chart(fig)

    del video_file, temp_video_path
    gc.collect()
