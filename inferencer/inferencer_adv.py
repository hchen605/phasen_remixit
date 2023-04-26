import os

import torch
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from inferencer.base_inferencer import BaseInferencer
from util.metrics import STOI, PESQ, mean_std
from model.base_SE_model import Base_SE_Model

from pesq import NoUtterancesError


def inference_wrapper(
    dataloader, model, device, inference_args, enhanced_dir
):
    result = {
        "filename": [], 
        "noisy_pesq": [], 
        "noisy_stoi": [],
        "enhanced_pesq": [], 
        "enhanced_stoi": [],
    }

    def add_fgsm_noise(model: Base_SE_Model, noisy: torch.Tensor, clean: torch.Tensor, eps=0.001):

        model.train()

        noisy.requires_grad = True

        loss = model(noisy, clean)
        noisy.retain_grad()
        loss.backward()
        noisy_grad = noisy.grad.data
        sign_grad = noisy_grad.sign()
        ratio = noisy.abs().max() * eps
        adv_wav = sign_grad * ratio + noisy

        model.eval()

        return adv_wav

    for noisy, clean, name in tqdm(dataloader, desc="Inference"):
        assert len(name) == 1, "The batch size of inference stage must 1."
        name = name[0]

        noisy = noisy.to(device)
        clean = clean.to(device)

        noisy = add_fgsm_noise(model, noisy, clean)

        with torch.no_grad():
            enhanced = model.inference(noisy)

        noisy = noisy.detach().squeeze().cpu().numpy()
        enhanced = enhanced.squeeze().cpu().numpy()
        clean = clean.squeeze().cpu().numpy()
        
        try:
            result["noisy_pesq"].append(PESQ(clean, noisy, sr=16000))
            result["enhanced_pesq"].append(PESQ(clean, enhanced, sr=16000))
        except NoUtterancesError:
            print("can't found utterence in {}! ignore it".format(name))
            continue

        result["filename"].append(name)
        result["noisy_stoi"].append(STOI(clean, noisy, sr=16000))
        result["enhanced_stoi"].append(STOI(clean, enhanced, sr=16000))


        sf.write(enhanced_dir / f"{name}_after.wav", enhanced, samplerate=16000)
        sf.write(enhanced_dir / f"{name}_before.wav", noisy, samplerate=16000)
        sf.write(enhanced_dir / f"{name}_clean.wav", clean, samplerate=16000)


    df = pd.DataFrame(result)
    print("NOISY PESQ: {:.4f} ± {:.4f}".format(*mean_std(df["noisy_pesq"].to_numpy())))
    print("NOISY STOI: {:.4f} ± {:.4f}".format(*mean_std(df["noisy_stoi"].to_numpy())))
    print("ENHANCED PESQ: {:.4f} ± {:.4f}".format(*mean_std(df["enhanced_pesq"].to_numpy())))
    print("ENHANCED STOI: {:.4f} ± {:.4f}".format(*mean_std(df["enhanced_stoi"].to_numpy())))

    df.to_csv(os.path.join(enhanced_dir, "_results.csv"), index=False)


class Inferencer(BaseInferencer):
    def __init__(self, config, checkpoint_path, output_dir):
        super(Inferencer, self).__init__(config, checkpoint_path, output_dir)

    def inference(self):
        inference_wrapper(
            dataloader=self.dataloader,
            model=self.model,
            device=self.device,
            inference_args=self.inference_config,
            enhanced_dir=self.enhanced_dir,
        )
