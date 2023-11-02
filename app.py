import streamlit as st
from st_audiorec import st_audiorec
import soundfile as sf
import librosa
import io
import os
import numpy as np
from utils.generator import CustomDataGenerator
from utils.inference import infer
def denoise(audio_data, sample_rate):
    # librosa
    # st.audio(audio, format="audio/wav")
    # librosa.load(audio)
    if sample_rate!=22050:
        audio_data, sample_rate = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=22050), 22050
    print(audio_data.shape, sample_rate)
    return CustomDataGenerator(audio_data, sample_rate, (200, 200)).get_image()
    

# Create a sidebar for navigation
st.sidebar.title("Audio App")
page = st.sidebar.radio("Go to", ["Record Sound", "Upload Audio"])

if not os.path.exists("audio"):
    os.mkdir("audio")  # Create an "audio" folder if it doesn't exist

if page == "Record Sound":
    st.title("Record Sound")
    
    # Generate a unique temporary audio file name in the "audio" folder
    # timestamp = dt.datetime.now().strftime(format="%Y-%m-%d %H-%M-%S")
    # temp_audio_file = os.path.join("audio", f"recording_{timestamp}.wav")

    wav_audio_data = st_audiorec()
    
elif page == "Upload Audio":
    st.title("Denoise Audio")
    
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])
    
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        with io.BytesIO(audio_bytes) as audio_io:
            audio_data, sample_rate = sf.read(audio_io)

            col1, col2 = st.columns(2)  # Create two columns

            with col1:
                img=denoise(audio_data.reshape(-1), sample_rate).numpy()
                st.image(img, caption='Noisy Image', use_column_width=True)
                

            with col2:
                preds=infer(img)
                # print(preds)
                st.image(preds, clamp=True, channels='RGB',  caption='Clean Image', use_column_width=True)
