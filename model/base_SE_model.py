import torch
import torch.nn as nn

class Base_SE_Model(nn.Module):
    def __init__(
        self,
        n_fft=512,
        win_length=512,
        hop_length=256,
        spec_factor=0.15,
        spec_abs_exponent=0.5,
        transform_type="none"
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = torch.hann_window(self.win_length).cuda()
        self.spec_factor = spec_factor
        self.spec_abs_exponent=spec_abs_exponent
        self.transform_type=transform_type

    def stft(self, wav):
        complex_spec = torch.stft(
            input=wav,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=self.window,
            normalized=False,
            return_complex=True,
        )
        complex_spec = self.spec_forward(complex_spec)

        return complex_spec

    def istft(self, complex_spec, length=None):
        complex_spec = self.spec_backward(complex_spec)
        wav = torch.istft(
            input=complex_spec,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=self.window,
            length=length,
        )
        return wav

    def convert_to_different_features(self, complex_spec):
        features = {}
        features["real"] = torch.real(complex_spec)
        features["imag"] = torch.imag(complex_spec)
        features["mag"] = torch.abs(complex_spec)
        features["phase"] = torch.angle(complex_spec)

        return features

    def spec_forward(self, spec: torch.Tensor):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs() ** e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor

        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_backward(self, spec:torch.Tensor):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs() ** (1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec
