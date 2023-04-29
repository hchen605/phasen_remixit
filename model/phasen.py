"""
yxhu@ASLP-NPU in Sogou inc.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from asteroid.losses import pairwise_neg_sisdr

from model.base_SE_model import Base_SE_Model


class FTB(nn.Module):
    def __init__(self, input_dim=257, in_channel=9, r_channel=5):

        super(FTB, self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, r_channel, kernel_size=[1, 1]),
            nn.BatchNorm2d(r_channel),
            nn.ReLU(),
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(r_channel * input_dim, in_channel, kernel_size=9, padding=4),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(),
        )
        self.freq_fc = nn.Linear(input_dim, input_dim, bias=False)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=[1, 1]),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
        )

    def forward(self, inputs):
        """
        inputs should be [Batch, Ca, Dim, Time]
        """
        # T-F attention
        conv1_out = self.conv1(inputs)
        B, C, D, T = conv1_out.size()
        reshape1_out = torch.reshape(conv1_out, [B, C * D, T])
        conv1d_out = self.conv1d(reshape1_out)
        conv1d_out = torch.reshape(conv1d_out, [B, self.in_channel, 1, T])

        # now is also [B,C,D,T]
        att_out = conv1d_out * inputs

        # tranpose to [B,C,T,D]
        att_out = torch.transpose(att_out, 2, 3)
        freqfc_out = self.freq_fc(att_out)
        att_out = torch.transpose(freqfc_out, 2, 3)

        cat_out = torch.cat([att_out, inputs], 1)
        outputs = self.conv2(cat_out)
        return outputs


class InforComu(nn.Module):
    def __init__(self, src_channel, tgt_channel):

        super(InforComu, self).__init__()
        self.comu_conv = nn.Conv2d(src_channel, tgt_channel, kernel_size=(1, 1))

    def forward(self, src, tgt):

        outputs = tgt * torch.tanh(self.comu_conv(src))
        return outputs


class GLayerNorm2d(nn.Module):
    def __init__(self, in_channel, eps=1e-12):
        super(GLayerNorm2d, self).__init__()
        self.eps = eps
        self.beta = nn.Parameter(torch.ones([1, in_channel, 1, 1]))
        self.gamma = nn.Parameter(torch.zeros([1, in_channel, 1, 1]))

    def forward(self, inputs):
        mean = torch.mean(inputs, [1, 2, 3], keepdim=True)
        var = torch.var(inputs, [1, 2, 3], keepdim=True)
        outputs = (inputs - mean) / torch.sqrt(var + self.eps) * self.beta + self.gamma
        return outputs


class TSB(nn.Module):
    def __init__(self, input_dim=257, channel_amp=9, channel_phase=8):
        super(TSB, self).__init__()

        self.ftb1 = FTB(
            input_dim=input_dim,
            in_channel=channel_amp,
        )
        self.amp_conv1 = nn.Sequential(
            nn.Conv2d(channel_amp, channel_amp, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU(),
        )
        self.amp_conv2 = nn.Sequential(
            nn.Conv2d(channel_amp, channel_amp, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU(),
        )
        self.amp_conv3 = nn.Sequential(
            nn.Conv2d(channel_amp, channel_amp, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU(),
        )

        self.ftb2 = FTB(
            input_dim=input_dim,
            in_channel=channel_amp,
        )

        self.phase_conv1 = nn.Sequential(
            nn.Conv2d(channel_phase, channel_phase, kernel_size=(5, 5), padding=(2, 2)),
            GLayerNorm2d(channel_phase),
        )
        self.phase_conv2 = nn.Sequential(
            nn.Conv2d(
                channel_phase, channel_phase, kernel_size=(1, 25), padding=(0, 12)
            ),
            GLayerNorm2d(channel_phase),
        )

        self.p2a_comu = InforComu(channel_phase, channel_amp)
        self.a2p_comu = InforComu(channel_amp, channel_phase)

    def forward(self, amp, phase):
        """
        amp should be [Batch, Ca, Dim, Time]
        amp should be [Batch, Cr, Dim, Time]

        """

        amp_out1 = self.ftb1(amp)
        amp_out2 = self.amp_conv1(amp_out1)
        amp_out3 = self.amp_conv2(amp_out2)
        amp_out4 = self.amp_conv3(amp_out3)
        amp_out5 = self.ftb2(amp_out4)

        phase_out1 = self.phase_conv1(phase)
        phase_out2 = self.phase_conv2(phase_out1)

        amp_out = self.p2a_comu(phase_out2, amp_out5)
        phase_out = self.a2p_comu(amp_out5, phase_out2)

        return amp_out, phase_out


class PHASEN(Base_SE_Model):
    def __init__(
        self,
        n_fft=512,
        win_length=400,
        hop_length=100,
        transform_type="none",
        num_blocks=3,
        channel_amp=24,
        channel_phase=12,
        loss_fn="TF-MSE",
        rnn_nums=300,
    ):
        super().__init__(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            transform_type=transform_type
        )
        self.num_blocks = 3
        self.feat_dim = n_fft // 2 + 1

        fix = True

        self.amp_conv1 = nn.Sequential(
            nn.Conv2d(2, channel_amp, kernel_size=[7, 1], padding=(3, 0)),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU(),
            nn.Conv2d(channel_amp, channel_amp, kernel_size=[1, 7], padding=(0, 3)),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU(),
        )
        self.phase_conv1 = nn.Sequential(
            nn.Conv2d(2, channel_phase, kernel_size=[3, 5], padding=(1, 2)),
            nn.Conv2d(
                channel_phase, channel_phase, kernel_size=[3, 25], padding=(1, 12)
            ),
        )

        self.tsbs = nn.ModuleList()
        for idx in range(self.num_blocks):
            self.tsbs.append(
                TSB(
                    input_dim=self.feat_dim,
                    channel_amp=channel_amp,
                    channel_phase=channel_phase,
                )
            )

        self.amp_conv2 = nn.Sequential(
            nn.Conv2d(channel_amp, 8, kernel_size=[1, 1]),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        phase_out_channels = 2

        self.phase_conv2 = nn.Sequential(
            nn.Conv2d(channel_phase, phase_out_channels, kernel_size=[1, 1])
        )
        self.rnn = nn.GRU(self.feat_dim * 8, rnn_nums, bidirectional=True)
        self.fcs = nn.Sequential(
                nn.Linear(rnn_nums * 2, 600),
                nn.ReLU(),
                nn.Linear(600, 600),
                nn.ReLU(),
                nn.Linear(600, self.feat_dim),
                nn.Sigmoid(),
        )

        if loss_fn not in ("TF-MSE"):
            raise NotImplementedError(
                "Loss function: {} is not implemented!".format(loss_fn)
            )
        self.loss_fn = loss_fn

    def _forward(self, noisy_wav):
        if noisy_wav.dim() == 2:
            complex_noisy_spec = self.stft(noisy_wav)
            noisy_features = self.convert_to_different_features(complex_noisy_spec)

            noisy_mag_spec = torch.unsqueeze(noisy_features["mag"], 1)
            complex_noisy_spec = torch.stack([noisy_features["real"], noisy_features["imag"]], dim=1)
        elif noisy_wav.dim() == 4:
            # the input has been converted to complex_spec
            complex_noisy_spec = noisy_wav
        else:
            raise ValueError

        masking_spec = self.amp_conv1(complex_noisy_spec)
        phase = self.phase_conv1(complex_noisy_spec)
        s_spec = masking_spec
        s_phase = phase
        for idx, layer in enumerate(self.tsbs):
            if idx != 0:
                masking_spec += s_spec
                phase += s_phase
            masking_spec, phase = layer(masking_spec, phase)
        masking_spec = self.amp_conv2(masking_spec)

        masking_spec = torch.transpose(masking_spec, 1, 3)
        B, T, D, C = masking_spec.size()
        masking_spec = torch.reshape(masking_spec, [B, T, D * C])
        masking_spec = self.rnn(masking_spec)[0]
        masking_spec = self.fcs(masking_spec)

        masking_spec = torch.reshape(masking_spec, [B, T, D, 1])
        masking_spec = torch.transpose(masking_spec, 1, 3)

        phase = self.phase_conv2(phase)

        return noisy_mag_spec, masking_spec, phase

    def forward(self, noisy_wav, clean_wav):
        noisy_mag_spec, masking_spec, phase = self._forward(noisy_wav)
        enhanced_mag_spec = noisy_mag_spec * masking_spec #[32, 1, 257, 188]
        
        if self.loss_fn == "TF-MSE":
            phase = phase / (
                torch.sqrt(torch.abs(phase[:, 0]) ** 2 + torch.abs(phase[:, 1]) ** 2)
                + 1e-8
            ).unsqueeze(1)
            enhanced_cspec = enhanced_mag_spec * phase
            complex_clean_spec = self.stft(clean_wav)
            clean_features = self.convert_to_different_features(complex_clean_spec)
            complex_clean_spec = torch.stack([clean_features["real"], clean_features["imag"]], dim=1)

            mag_loss = F.mse_loss(clean_features["mag"], enhanced_mag_spec.squeeze(1))
            phase_loss = F.mse_loss(complex_clean_spec, enhanced_cspec) / 2
            all_loss = mag_loss + phase_loss

            # add SI-SDR
            complex_enhanced_spec = enhanced_mag_spec * phase #[1, 2, 257, T]

            complex_enhanced_spec = torch.complex(real=complex_enhanced_spec[:, 0], imag=complex_enhanced_spec[:, 1])
            #[1, 257, T]
            enhanced_wav = self.istft(complex_enhanced_spec, length=noisy_wav.size(1))
            noise_wav = noisy_wav - enhanced_wav
            enhanced_wav = enhanced_wav[:, None, :]
            clean_wav = clean_wav[:, None, :]
            loss_val = pairwise_neg_sisdr(enhanced_wav, clean_wav)

            all_loss = all_loss + torch.mean(loss_val)

            return all_loss
        else:
            raise NotImplementedError

    def inference(self, noisy_wav):
        noisy_mag_spec, masking_spec, phase = self._forward(noisy_wav)
        enhanced_mag_spec = noisy_mag_spec * masking_spec

        phase = phase / (
            torch.sqrt(torch.abs(phase[:, 0]) ** 2 + torch.abs(phase[:, 1]) ** 2)
            + 1e-8
        ).unsqueeze(1)

        complex_enhanced_spec = enhanced_mag_spec * phase #[1, 2, 257, T]

        complex_enhanced_spec = torch.complex(real=complex_enhanced_spec[:, 0], imag=complex_enhanced_spec[:, 1])
        #[1, 257, T]
        enhanced_wav = self.istft(complex_enhanced_spec, length=noisy_wav.size(1))
        noise_wav = noisy_wav - enhanced_wav

        return enhanced_wav, noise_wav

        
    def inference_wiener(self, noisy_wav):
        noisy_mag_spec, masking_spec, phase = self._forward(noisy_wav)
        enhanced_mag_spec = noisy_mag_spec * masking_spec #[1, 1, 257, T]
        noise_mag_spec = noisy_mag_spec - enhanced_mag_spec

        phase = phase / (
            torch.sqrt(torch.abs(phase[:, 0]) ** 2 + torch.abs(phase[:, 1]) ** 2)
            + 1e-8
        ).unsqueeze(1)
        
        # wiener filter 1
        alpha = 1.05
        #enhanced_amp = torch.abs(enhanced_mag_spec)
        #noisy_amp = torch.abs(noisy_mag_spec)
        noisy_mag_spec = torch.add(noisy_mag_spec, 0.001)
        gain = torch.pow(torch.div(enhanced_mag_spec, noisy_mag_spec), alpha)
        #gain = torch.div(torch.pow(enhanced_mag_spec, alpha), (torch.pow(enhanced_mag_spec, alpha) + torch.pow(noise_mag_spec, alpha)))
        #print(torch.nonzero(torch.isnan(gain)))
        #gain = torch.nan_to_num(gain)
        enhanced_mag_spec = torch.mul(noisy_mag_spec, gain) #[1, 1, 257, T]
        #

        complex_enhanced_spec = enhanced_mag_spec * phase #[1, 2, 257, T]
        #complex_noisy_spec = noisy_mag_spec * phase #[1, 2, 257, T]

        complex_enhanced_spec = torch.complex(real=complex_enhanced_spec[:, 0], imag=complex_enhanced_spec[:, 1])#[1, 257, T]
        #complex_noisy_spec = torch.complex(real=complex_noisy_spec[:, 0], imag=complex_noisy_spec[:, 1]) #[1, 257, T]
        
        #
        enhanced_wav = self.istft(complex_enhanced_spec, length=noisy_wav.size(1))
        
        '''
        ## wiener filter 2
        noise_psd = torch.mean(noisy_mag_spec - enhanced_mag_spec, 3, True)
        clean_psd = torch.mean(enhanced_mag_spec, 3, True)
        wiener_filter = torch.divide(clean_psd, (clean_psd + noise_psd))
        #filtered_data = signal.lfilter(wiener_filter, 1, noisy_data)
        enhanced_mag_spec = torch.mul(noisy_mag_spec, wiener_filter)
        complex_enhanced_spec = enhanced_mag_spec * phase #[1, 2, 257, T]
        complex_enhanced_spec = torch.complex(real=complex_enhanced_spec[:, 0], imag=complex_enhanced_spec[:, 1])#[1, 257, T]
        
        #
        enhanced_wav = self.istft(complex_enhanced_spec, length=noisy_wav.size(1))
        '''

        return enhanced_wav
