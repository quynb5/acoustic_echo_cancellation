import io
import numpy as np
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt

from scipy.signal import welch
from .model import acoustic_echo_cancellation

# Hàm vẽ biểu đồ dạng sóng của tín hiệu âm thanh
def plot_waveform(audio, samplerate, title):
    times = np.linspace(0, len(audio) / samplerate, num=len(audio))  # Tính trục thời gian
    plt.figure(figsize=(10, 5))
    plt.plot(times, audio, color='black')  # Vẽ tín hiệu âm thanh
    plt.xlabel("Time (secs)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid()
    st.pyplot(plt)
    
# Hàm vẽ phổ tần số của tín hiệu âm thanh
def plot_frequency_response(audio, samplerate, title):
    freqs, psd = welch(audio, samplerate, nperseg=1024)
    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, psd)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title(title)
    plt.grid()
    st.pyplot(plt)

# Hàm chính để chạy ứng dụng Streamlit
def app(model):
    st.set_page_config(layout="wide")

    st.title("🔊 AEC (Acoustic Echo Cancellation) Demo")

    # Giao diện upload file âm thanh
    col1, col2, _ = st.columns([1, 1, 1])
    with col1:
        st.subheader("📂 Upload Mic Audio File")
        uploaded_mic_audio = st.file_uploader("Upload mic audio file", type=["wav", "mp3"], label_visibility='collapsed', key="file1")

    with col2:
        st.subheader("📂 Upload Ref Audio File")
        uploaded_ref_audio = st.file_uploader("Upload ref audio file", type=["wav", "mp3"], label_visibility='collapsed', key="file2")

    # Nút nhấn xử lý âm thanh
    st.subheader("🛠️ Process Audio")
    # Tạo giao diện nút xử lý âm thanh
    st.markdown("""
    <style>
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    process_clicked = st.button("Process Audio")

    # Biến trạng thái để kiểm tra đã xử lý hay chưa
    if 'processed' not in st.session_state:
        st.session_state.processed = False

    # Xử lý âm thanh nếu nút được nhấn
    if process_clicked:
        # Kiểm tra xem đã upload cả hai file âm thanh chưa
        if uploaded_mic_audio is not None and uploaded_ref_audio is not None:
            # Đọc file âm thanh
            mic_audio, mic_samplerate = sf.read(uploaded_mic_audio)
            ref_audio, ref_samplerate = sf.read(uploaded_ref_audio)
            
            # Kiểm tra xem hai file có cùng tần số lấy mẫu không
            if mic_samplerate != ref_samplerate:
                # Thông báo lỗi nếu tần số lấy mẫu không khớp
                st.error("Lỗi: Tần số lấy mẫu của hai file âm thanh không khớp.")
            else:
                # Xử lý AEC (Loại bỏ tiếng vọng)
                processed_audio = acoustic_echo_cancellation(
                    model, mic_audio, ref_audio, mic_samplerate, align=False
                )
                
                # Lưu file xử lý vào buffer
                buffer = io.BytesIO()
                sf.write(buffer, processed_audio, mic_samplerate, format='WAV')
                buffer.seek(0)
                
                # Cập nhật các trạng thái
                st.session_state.processed = True
                st.session_state.mic_audio = mic_audio
                st.session_state.ref_audio = ref_audio
                st.session_state.processed_audio = processed_audio
                st.session_state.samplerate = mic_samplerate
        else:
            # Thông báo lỗi nếu chưa tải lên cả hai file âm thanh
            st.warning("⚠️ Vui lòng tải lên cả hai file âm thanh trước khi xử lý.")

    # Kiểm tra xem đã xử lý âm thanh chưa
    if st.session_state.processed:
        # Tạo tiêu đề và hiển thị biểu đồ sóng của tín hiệu âm thanh
        st.subheader("📊 AEC Frequency Spectrum")
        col3, col4, col5 = st.columns(3)
        with col3:
            # Hiển thị biểu đồ sóng của tín hiệu âm thanh
            plot_waveform(st.session_state.mic_audio, st.session_state.samplerate, "Mic Audio Waveform")
        with col4:
            plot_waveform(st.session_state.ref_audio, st.session_state.samplerate, "Ref Audio Waveform")
        with col5:
            plot_waveform(st.session_state.processed_audio, st.session_state.samplerate, "Processed Audio Waveform")
        
        # Hiển thị và cho phép tải xuống file âm thanh đã xử lý
        buffer = io.BytesIO()
        sf.write(buffer, st.session_state.processed_audio, st.session_state.samplerate, format='WAV')
        buffer.seek(0)
        st.audio(buffer, format='audio/wav')
        # Tạo nút tải xuống file âm thanh đã xử lý
        st.download_button("⬇️ Download Processed Audio", buffer, file_name="processed_audio.wav", mime="audio/wav")
    else:
        # Hiển thị biểu đồ phổ tần số trống ban đầu
        st.subheader("📊 Frequency Spectrum")
        col1, col2, col3 = st.columns(3)
        with col1:
            plot_frequency_response(np.zeros(1024), 44100, "Mic Audio Spectrum")
        with col2:
            plot_frequency_response(np.zeros(1024), 44100, "Ref Audio Spectrum")
        with col3:
            plot_frequency_response(np.zeros(1024), 44100, "Processed Audio Spectrum")
