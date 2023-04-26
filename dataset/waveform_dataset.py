import os

import torch
import librosa
import numpy as np

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

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        noisy_path, clean_path = self.dataset_list[item].split(" ")
        name = os.path.splitext(os.path.basename(noisy_path))[0]

        #name edit for mic
        #name = os.path.splitext(noisy_path)[0]
        #print(name)
        '''
        if 'crisp' in name:
            name = name.split("/")
            name = name[6]+'_'+name[10]+'_'+os.path.splitext(os.path.basename(noisy_path))[0]
        else:
            name = name.split("/")
            name = name[6]+'_'+name[7]+'_'+os.path.splitext(os.path.basename(noisy_path))[0]
        '''
        #

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

        return noisy, clean, name

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
