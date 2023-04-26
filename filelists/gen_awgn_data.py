import os
import numpy as np
import librosa
import random
import soundfile as sound
from tqdm import tqdm

def adjust_ratio_on_SNR(clean_wav, noise_wav, SNR):
    signal_energy = np.mean(clean_wav * clean_wav)
    noise_energy = np.mean(noise_wav * noise_wav)

    ### Convert dB to linear scale
    SNR = 10 ** (SNR / 10)

    return np.sqrt(signal_energy / (noise_energy * SNR))

def cut_or_pad(noise, target_length):
    noise_length = len(noise)
    return_noise = np.zeros(target_length)
    if noise_length >= target_length:
        return_noise = noise[:target_length]
    elif noise_length < target_length:
        start = random.choice(range(target_length - noise_length))
        return_noise[start : start + noise_length] = noise

    return return_noise

def additive_noise_simulation(input_wav, SNR_value):

    noise = np.random.normal(size=input_wav.shape)
    noise = cut_or_pad(noise, len(input_wav))
    ratio = adjust_ratio_on_SNR(input_wav, noise, SNR_value)

    return input_wav + noise * ratio


test_clean_dir = '/home/koredata/hsinhung/speech/vb_demand/clean_testset_wav'
test_awgn_dir = '/home/koredata/hsinhung/speech/vb_demand/testset_awgn/snr_20'
snr = 20
fs = 16000

if not os.path.exists(test_awgn_dir):
    os.makedirs(test_awgn_dir)

#for path in test_clean_dir:
for (dirpath, dirnames, filenames) in os.walk(test_clean_dir):
    print(dirpath)
    #print(len(filenames))
    for f in tqdm(filenames):
        if not f.endswith((".WAV", ".wav")):
            continue
            
        noisy_path = os.path.join(test_awgn_dir, f)
        clean_path = os.path.join(test_clean_dir, f)

        clean, _ = librosa.load(clean_path, sr=fs)
        noisy = additive_noise_simulation(clean, snr)
        
        #print(f)
        sound.write(noisy_path, noisy, fs)

