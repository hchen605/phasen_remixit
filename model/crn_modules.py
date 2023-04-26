import torch.nn as nn
import torch.nn.functional as F



class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 2),
            stride=(2, 1),
            padding=(2, 1),
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class CausalTransConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dropout=0.0,
        is_last=False,
        padding=(0, 0),
        output_padding=(0, 0),
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 2),
            stride=(2, 1),
            padding=padding,
            output_padding=output_padding,
        )
        self.is_last = is_last
        if not is_last:
            self.norm = nn.BatchNorm2d(num_features=out_channels)
            self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        if not self.is_last:
            x = self.norm(x)
            x = self.activation(x)
        
        x = self.dropout(x)
        return x


class CNN_Encoder(nn.Module):
    def __init__(
        self,
        num_layers=6,
        channels_list=[1, 16, 32, 64, 128, 256, 256],
        kernel_size=5,
        dropout=0.0,
        grad_stopping=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        for idx in range(self.num_layers):
            in_channels = channels_list[idx]
            out_channels = channels_list[idx + 1]
            self.layers.append(
                CausalConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
        self.grad_stopping = grad_stopping

    def forward(self, inputs):
        representation_list = []
        for idx in range(self.num_layers):
            outputs = self.layers[idx](inputs)
            representation_list.append(outputs)
            inputs = outputs.detach() if self.grad_stopping else outputs

        return representation_list
