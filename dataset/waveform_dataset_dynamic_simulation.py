import os
import random

import torch
import librosa
import numpy as np
import scipy.signal as ss

from torch.utils import data

from util.utils import sample_fixed_length_data_aligned


class Dataset(data.Dataset):
    def __init__(
        self,
        dataset_list,
        limit,
        offset,
        sample_rate,
        is_training,
        sample_length=48000,
        max_length=None,
        do_normalize=True,
        additive_noise_list=None,
        SNR_values=[5, 10, 15],
        rir_list=None,
    ):
        """
        dataset_list(*.txt):
            <noisy_path> <clean_path>\n
        e.g:
            noisy_1.wav clean_1.wav
            noisy_2.wav clean_2.wav
            ...
            noisy_n.wav clean_n.wav
        """
        super(Dataset, self).__init__()
        self.sample_rate = sample_rate
        self.is_training = is_training

        dataset_list = [line.rstrip("\n") for line in open(dataset_list, "r")]
        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        self.dataset_list = dataset_list
        self.length = len(self.dataset_list)
        self.do_normalize = do_normalize
        self.sample_length = sample_length
        self.max_length = max_length

        if additive_noise_list:
            self.additive_noise_list = [
                line.rstrip("\n") for line in open(additive_noise_list, "r")
            ]
            self.SNR_values = SNR_values
        else:
            self.additive_noise_list = None
            self.SNR_values = SNR_values

        if rir_list:
            self.rir_list = [line.rstrip("\n") for line in open(rir_list, "r")]
            self.predelay_sec = 0.05
        else:
            self.rir_list = None

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        noisy_path, clean_path = self.dataset_list[item].split(" ")
        name = os.path.splitext(os.path.basename(noisy_path))[0]
        noisy, _ = librosa.load(noisy_path, sr=self.sample_rate)
        clean, _ = librosa.load(clean_path, sr=self.sample_rate)

        if self.do_normalize:
            noisy = self.normalize_wav(noisy)
            clean = self.normalize_wav(clean)

        if self.is_training:
            # The input of model should be fixed-length in the training.
            noisy, clean = sample_fixed_length_data_aligned(
                noisy, clean, self.sample_length
            )
        elif self.max_length:
            # This is for SaShiMi validation to avoid OOM
            if len(noisy) > self.max_length:
                noisy = noisy[:self.max_length]
                clean = clean[:self.max_length]

        if noisy_path == clean_path:
            #random_value = random.random()
            #if random_value < 0.8:
            noisy = self.additive_noise_simulation(noisy)
            #if random_value > 0.5:
            #    noisy, clean = self.reverberant_noise_simulation(noisy, clean)
        else:
            random_value = random.random()
            if random_value < 0.5:
                noisy = self.additive_noise_simulation(noisy)

        noisy = noisy.astype("float32")
        clean = clean.astype("float32")

        return noisy, clean, name

    def additive_noise_simulation(self, input_wav):
        if not self.additive_noise_list:
            noise = np.random.normal(size=input_wav.shape)
        else:
            noise_path = random.choice(self.additive_noise_list)
            noise, _ = librosa.load(noise_path, sr=self.sample_rate)

        noise = cut_or_pad(noise, len(input_wav))
        SNR_value = random.choice(self.SNR_values)
        ratio = adjust_ratio_on_SNR(input_wav, noise, SNR_value)
        return input_wav + noise * ratio

    def reverberant_noise_simulation(self, input_wav, clean_wav):
        if not self.rir_list:
            return input_wav, clean_wav

        rir_path = random.choice(self.rir_list)
        rir, _ = librosa.load(rir_path, sr=self.sample_rate)
        rir = rir * 10
        dt = np.argmax(rir).min()
        et = dt + int(self.predelay_sec * self.sample_rate)
        rir_direct = rir[:et]

        noisy_wav = ss.convolve(input_wav, rir)
        target_wav = ss.convolve(clean_wav, rir_direct)

        return noisy_wav[: len(input_wav)], target_wav[: len(input_wav)]

    def normalize_wav(self, wav):
        return wav / np.abs(wav).max()


def collate_fn(batch):
    noisy_list = []
    clean_list = []
    names = []

    for noisy, clean, name in batch:
        noisy_list.append(torch.tensor(noisy))  # [F, T] => [T, F]
        clean_list.append(torch.tensor(clean))  # [1, T] => [T, 1]
        names.append(name)

    noisy_wav = torch.stack(noisy_list, dim=0)
    clean_wav = torch.stack(clean_list, dim=0)

    return noisy_wav, clean_wav, names


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
