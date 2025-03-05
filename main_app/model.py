import torch

from .nkf_aec import NKF
from .utils import gcc_phat


# Khởi tạo mô hình xử lý tiếng vọng
def init_model(weights_path='resources/weights/nkf_epoch70.pt'):
    model = NKF(L=4)
    numparams = 0
    for f in model.parameters():
        numparams += f.numel()
    print('Total number of parameters: {:,}'.format(numparams))
    model.load_state_dict(torch.load(weights_path), strict=True)
    model.eval()
    
    return model


# Hàm xử lý tiếng vọng
def acoustic_echo_cancellation(model, mic_audio, ref_audio, samplerate, align=False):
    # Chuyển dữ liệu âm thanh về dạng tensor
    mic_audio = torch.from_numpy(mic_audio).float()  
    ref_audio = torch.from_numpy(ref_audio).float()

    # Cân chỉnh độ trễ giữa hai tín hiệu âm thanh nếu cần
    if align:
        tau = gcc_phat(mic_audio[:samplerate * 10], ref_audio[:samplerate * 10], fs=samplerate, interp=1)
        tau = max(0, int((tau - 0.001) * samplerate))
        ref_audio = torch.cat([torch.zeros(tau), ref_audio])[:mic_audio.shape[-1]]

    # Xử lý tiếng vọng
    with torch.no_grad():
        result_hat = model(ref_audio, mic_audio)

    # Trả về tín hiệu âm thanh đã xử lý
    return result_hat.cpu().numpy()
