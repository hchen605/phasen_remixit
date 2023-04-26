import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_SE_model import Base_SE_Model
from model.crn_modules import CNN_Encoder, CausalConvBlock, CausalTransConvBlock


class CRN_parent(Base_SE_Model):
    def __init__(
        self,
        cnn_layers=6,
        kernel_size=5,
        channels=[16, 32, 64, 128, 256, 256],
        complex_feature=False,
        loss_fn="TF-MSE",
        output_scenario="regression",
        n_fft=512,
        win_length=512,
        hop_length=256,
        spec_factor=0.15,
        spec_abs_exponent=0.5,
        transform_type="exponent",
        grad_stopping=False,
    ):
        super().__init__(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            spec_factor=spec_factor,
            spec_abs_exponent=spec_abs_exponent,
            transform_type=transform_type
        )

        self.complex_feature = complex_feature
        input_channels = 2 if complex_feature else 1
        self.cnn_layers = cnn_layers

        self.input_encoder = CNN_Encoder(  # f_i in AL
            num_layers=self.cnn_layers,
            channels_list=[input_channels] + channels,
            kernel_size=kernel_size,
            grad_stopping=grad_stopping,
        )
        self.input_norm_layer = nn.BatchNorm2d(input_channels)

        self.decoder = (
            nn.ModuleList()
        )  # the decoder in BP model is also the h-function in AL model
        for idx in range(self.cnn_layers):
            if idx == 0:
                dec_out_channels = 2 if complex_feature else 1
            else:
                dec_out_channels = channels[idx - 1]

            dec_in_channels = channels[idx]

            is_last = idx == 0
            self.decoder.append(
                CausalTransConvBlock(
                    in_channels=2 * dec_in_channels,  # times 2 due to skip connection
                    out_channels=dec_out_channels,
                    kernel_size=kernel_size,
                    padding=(2, 0),
                    is_last=is_last,
                )
            )

        self.input_dim_list = []
        input_dim = self.n_fft // 2 + 1
        for channel in channels:
            input_dim = (input_dim + 4 - kernel_size) // 2 + 1
            self.input_dim_list.append(input_dim * channel)

        if loss_fn not in ("TF-MSE", "T-MSE", "SI-SNR", "TF+T-MSE"):
            raise NotImplementedError(
                "Loss function: {} is not implemented!".format(loss_fn)
            )

        self.loss_fn = loss_fn

        if output_scenario not in ("masking", "regression"):
            raise NotImplementedError(
                "Output scenario should be either 'masking' or 'regression'!"
            )
        self.output_scenario = output_scenario


    def apply_masking(self, dec_outputs, noisy_mag, noisy_phase):

        if self.complex_feature:
            assert dec_outputs.size(1) == 2
            mask_real = dec_outputs[:, 0]
            mask_imag = dec_outputs[:, 1]
            mask_mag = torch.sqrt(mask_real**2 + mask_imag**2)
            mask_mag = torch.tanh(mask_mag)
            mask_phase = torch.atan2(mask_imag, mask_real)

            enhanced_mag = noisy_mag * mask_mag
            enhanced_phase = noisy_phase + mask_phase
            enhanced_real = enhanced_mag * torch.cos(enhanced_phase)
            enhanced_imag = enhanced_mag * torch.sin(enhanced_phase)
            return torch.stack([enhanced_real, enhanced_imag], dim=1)  # [B, 2, F, T]
        else:
            assert dec_outputs.size(1) == 1
            mask_mag = torch.sigmoid(dec_outputs)
            enhanced_mag = noisy_mag.unsqueeze(1) * mask_mag
            return enhanced_mag


class BP_CRN(CRN_parent):
    def __init__(
        self,
        cnn_layers=6,
        kernel_size=5,
        channels=[16, 32, 64, 128, 256, 256],
        rnn_layers=2,
        rnn_units=128,
        bidirectional=False,
        complex_feature=False,
        loss_fn="TF_MSE",
        output_scenario="regression",
        n_fft=512,
        win_length=512,
        hop_length=256,
        transform_type="exponent"
    ):
        super().__init__(
            cnn_layers=cnn_layers,
            kernel_size=kernel_size,
            channels=channels,
            complex_feature=complex_feature,
            loss_fn=loss_fn,
            output_scenario=output_scenario,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            transform_type=transform_type,
            grad_stopping=False,
        )

        # LSTM
        self.lstm_prenet = nn.Linear(self.input_dim_list[-1], rnn_units)
        self.lstm = nn.LSTM(
            input_size=rnn_units,
            hidden_size=rnn_units,
            num_layers=rnn_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.lstm_postnet = nn.Linear(rnn_units, self.input_dim_list[-1])

    def _forward(self, noisy_wav):
        if noisy_wav.dim() == 2:
            complex_noisy_spec = self.stft(noisy_wav)
        elif noisy_wav.dim() == 3:
            complex_noisy_spec = noisy_wav
        else:
            raise ValueError

        noisy_features = self.convert_to_different_features(complex_noisy_spec)
        if self.complex_feature:
            inputs = torch.stack(
                [noisy_features["real"], noisy_features["imag"]], dim=1
            )
        else:
            inputs = noisy_features["mag"].unsqueeze(1)

        inputs = self.input_norm_layer(inputs)
        s_list = self.input_encoder(inputs)
        enc_outputs = s_list[-1]  # in BP we only needs the last output

        batch_size, n_channels, n_f_bins, n_frame_size = enc_outputs.shape
        lstm_inputs = enc_outputs.reshape(
            batch_size, n_channels * n_f_bins, n_frame_size
        ).permute(
            0, 2, 1
        )  # [Batch, time, channels]
        lstm_inputs = self.lstm_prenet(lstm_inputs)
        lstm_outputs, _ = self.lstm(lstm_inputs)
        lstm_outputs = self.lstm_postnet(lstm_outputs)
        lstm_outputs = lstm_outputs.permute(0, 2, 1).reshape(
            batch_size, n_channels, n_f_bins, n_frame_size
        )

        dec_outputs = lstm_outputs
        for idx in reversed(range(self.cnn_layers)):
            dec_inputs = torch.concat([s_list[idx], dec_outputs], dim=1)
            dec_outputs = self.decoder[idx](dec_inputs)

        if self.output_scenario == "masking":
            dec_outputs = self.apply_masking(
                dec_outputs, noisy_features["mag"], noisy_features["phase"]
            )
        elif not self.complex_feature:
            dec_outputs = F.relu(
                dec_outputs
            )  # Output is magnitude and thus should be positive

        return dec_outputs, noisy_features

    def forward(self, noisy_wav, clean_wav):
        self.lstm.flatten_parameters()
        dec_outputs, _ = self._forward(noisy_wav)

        if self.loss_fn == "TF-MSE":
            complex_clean_spec = self.stft(clean_wav)
            clean_feautures = self.convert_to_different_features(complex_clean_spec)
            if self.complex_feature:
                ground_truths = torch.stack(
                    [clean_feautures["real"], clean_feautures["imag"]], dim=1
                )
            else:
                ground_truths = clean_feautures["mag"].unsqueeze(1)

            return F.mse_loss(dec_outputs, ground_truths)
        else:
            raise NotImplementedError(
                "Loss function: {} is not implemented!".format(self.loss_fn)
            )

    def inference(self, noisy_wav):
        dec_outputs, noisy_features = self._forward(noisy_wav)

        if self.complex_feature:
            enhanced_real = dec_outputs[:, 0]
            enhanced_imag = dec_outputs[:, 1]
        else:
            enhanced_real = dec_outputs.squeeze(1) * torch.cos(noisy_features["phase"])
            enhanced_imag = dec_outputs.squeeze(1) * torch.sin(noisy_features["phase"])
        complex_enhanced_spec = torch.complex(real=enhanced_real, imag=enhanced_imag)
        enhanced_wav = self.istft(complex_enhanced_spec, length=noisy_wav.size(1))
        return enhanced_wav


