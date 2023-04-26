import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import copy

from trainer.base_trainer import BaseTrainer

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(self, config, resume: bool, teacher_model, student_model, optimizer, scheduler, train_dataloader, validation_dataloader):
        super(Trainer, self).__init__(config, resume, student_model, optimizer, scheduler)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.teacher_model = teacher_model
        self.teacher_model = teacher_model.to(self.device)
        self._pretrained_teacher(config)
        self.batch_size = config["train_dataloader"]["batch_size"]

    def _pretrained_teacher(self, config):
        if config["preloaded_model_path"]:
             model_path = Path(config["preloaded_model_path"])

        model_path = model_path.expanduser().absolute()
        assert model_path.exists(), f"Preloaded *.pth file is not exist. Please check the file path: {model_path.as_posix()}"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=self.device)

        if isinstance(self.model, torch.nn.DataParallel):
            self.teacher_model.module.load_state_dict(model_checkpoint, strict=False)
        else:
            self.teacher_model.load_state_dict(model_checkpoint, strict=False)

        print(f"Teacher Model preloaded successfully from {model_path.as_posix()}.")
        pass

    def _train_epoch(self, epoch):
        loss_total = 0.0

        pbar = tqdm(self.train_dataloader)
        for noisy, clean, name in pbar:
            self.optimizer.zero_grad()

            noisy = noisy.to(self.device)  # [Batch, length]
            clean = clean.to(self.device)  # [Batch, length]

            with torch.no_grad():
                # Teacher's estimates
                t_est_speech, t_est_noise = self.teacher_model.inference(noisy) #[32, 48000]
                # remix
                # Bootstrapped remixing
                permuted_t_est_noise = t_est_noise[torch.randperm(len(t_est_speech))]
                bootstrapped_mix = t_est_speech + permuted_t_est_noise
                #print('bootstrapped_mix: ', bootstrapped_mix.size())

            loss = self.model(bootstrapped_mix, t_est_speech)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            loss_total += loss.item()
            pbar.set_description("Loss: {:.3f}".format(loss.item()))

            # ema
            t_momentum = 0.99
            if epoch > 0:
                new_teacher_w = copy.deepcopy(self.teacher_model.state_dict())
                student_w = self.model.state_dict()
                for key in new_teacher_w.keys():
                    new_teacher_w[key] = (
                            t_momentum * new_teacher_w[key] + (1.0 - t_momentum) * student_w[key])

                self.teacher_model.load_state_dict(new_teacher_w)
                del new_teacher_w

        self.writer.add_scalar(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        noisy_list = []
        clean_list = []
        enhanced_list = []

        loss_total = 0.0

        visualization_limit = self.validation_custom_config["visualization_limit"]


        for i, (noisy, clean, name) in tqdm(enumerate(self.validation_dataloader), desc="Inference"):
            assert len(name) == 1, "The batch size of inference stage must 1."
            name = name[0]

            if noisy.size(1) < 16000*0.6:
                print(f"Warning! {name} is too short for computing STOI. Will skip this for now.")
                continue

            noisy = noisy.to(self.device)  # [Batch, length]
            clean = clean.to(self.device)  # [Batch, length]

            loss = self.model(noisy, clean)
            enhanced, _ = self.model.inference(noisy)

            loss_total += loss.item()
            noisy = noisy.squeeze(0).cpu().numpy()
            enhanced = enhanced.squeeze(0).cpu().numpy() # remove the batch dimension
            clean = clean.squeeze(0).cpu().numpy()

            assert len(noisy) == len(clean) == len(enhanced)

            if i <= np.min([visualization_limit, len(self.validation_dataloader)]):
                self.spec_audio_visualization(noisy, enhanced, clean, name, epoch)

            noisy_list.append(noisy)
            clean_list.append(clean)
            enhanced_list.append(enhanced)

        self.writer.add_scalar(f"Loss/Validation", loss_total / len(self.validation_dataloader), epoch)
        return self.metrics_visualization(noisy_list, clean_list, enhanced_list, epoch)
