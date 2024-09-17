import torch
from .base_vae import BaseVAE
from .cnn_size import CNNForwardSize, CNNBackwardSize


class CNNVAE(BaseVAE):
    def __init__(
            self,
            image_length: int,
            latent_dim: int,
            is_grayscale: bool = False,
            activate_func: str = "sigmoid",
            use_kl_mc: bool = True,
            lr: float = 1e-3,
            cnn_padding: int = 1,
            cnn_dilation: int = 1,
            cnn_kernel_size: int = 3,
            cnn_stride: int = 1,
    ):
        super().__init__(use_kl_mc=use_kl_mc, lr=lr)
        self.input_channel_num = 1 if is_grayscale else 3
        self.image_length = image_length
        self.latent_dim = latent_dim

        self.cnn_padding = cnn_padding
        self.cnn_dilation = cnn_dilation
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_stride = cnn_stride
        self.cnn_forward_size = CNNForwardSize(
            padding=self.cnn_padding,
            dilation=self.cnn_dilation,
            kernel_size=self.cnn_kernel_size,
            stride=self.cnn_stride,
        )
        self.cnn_backward_size = CNNBackwardSize(
            padding=self.cnn_padding,
            dilation=self.cnn_dilation,
            kernel_size=self.cnn_kernel_size,
            stride=self.cnn_stride,
        )

        if activate_func == "tanh":
            self.activate_func = torch.nn.Tanh()
        elif activate_func == "sigmoid":
            self.activate_func = torch.nn.Sigmoid()
        elif activate_func == "relu":
            self.activate_func = torch.nn.ReLU()
        else:
            raise RuntimeError(
                f"Error! Unknown value for 'activate_func': {activate_func}. Only allowing for 'tanh' and 'sigmoid'."
            )

        self.set_encoder()
        self.set_fc_mu()
        self.set_fc_log_var()
        self.set_decoder()

    def set_encoder(self):
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.input_channel_num,
                32,
                kernel_size=(self.cnn_kernel_size, self.cnn_kernel_size),
                stride=(self.cnn_stride, self.cnn_stride),
                padding=(self.cnn_padding, self.cnn_padding),
                dilation=(self.cnn_dilation, self.cnn_dilation),
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                32,
                64,
                kernel_size=(self.cnn_kernel_size, self.cnn_kernel_size),
                stride=(self.cnn_stride, self.cnn_stride),
                padding=(self.cnn_padding, self.cnn_padding),
                dilation=(self.cnn_dilation, self.cnn_dilation),
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                64,
                128,
                kernel_size=(self.cnn_kernel_size, self.cnn_kernel_size),
                stride=(self.cnn_stride, self.cnn_stride),
                padding=(self.cnn_padding, self.cnn_padding),
                dilation=(self.cnn_dilation, self.cnn_dilation),
            ),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        self.cnn_forward_size(self.image_length, depth=3)
        self.cnn_backward_size(list(reversed(self.cnn_forward_size.get_size())))

    def set_fc_mu(self):
        self.fc_mu = torch.nn.Linear(
            128 * self.cnn_forward_size.get_size(2) ** 2,
            self.latent_dim,
        )

    def set_fc_log_var(self):
        self.fc_log_var = torch.nn.Linear(
            128 * self.cnn_forward_size.get_size(2) ** 2,
            self.latent_dim,
        )

    def set_decoder(self):
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(
                self.latent_dim,
                128 * self.cnn_forward_size.get_size(2) ** 2,
            ),
            torch.nn.Unflatten(
                1,
                (
                    128,
                    self.cnn_forward_size.get_size(2),
                    self.cnn_forward_size.get_size(2),
                ),
            ),
            torch.nn.ConvTranspose2d(
                128,
                64,
                kernel_size=(self.cnn_kernel_size, self.cnn_kernel_size),
                stride=(self.cnn_stride, self.cnn_stride),
                padding=(self.cnn_padding, self.cnn_padding),
                output_padding=(
                    self.cnn_backward_size.get_size(0),
                    self.cnn_backward_size.get_size(0),
                ),
                dilation=(self.cnn_dilation, self.cnn_dilation),
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                64,
                32,
                kernel_size=(self.cnn_kernel_size, self.cnn_kernel_size),
                stride=(self.cnn_stride, self.cnn_stride),
                padding=(self.cnn_padding, self.cnn_padding),
                output_padding=(
                    self.cnn_backward_size.get_size(1),
                    self.cnn_backward_size.get_size(1),
                ),
                dilation=(self.cnn_dilation, self.cnn_dilation),
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                32,
                self.input_channel_num,
                kernel_size=(self.cnn_kernel_size, self.cnn_kernel_size),
                stride=(self.cnn_stride, self.cnn_stride),
                padding=(self.cnn_padding, self.cnn_padding),
                output_padding=(
                    self.cnn_backward_size.get_size(2),
                    self.cnn_backward_size.get_size(2),
                ),
                dilation=(self.cnn_dilation, self.cnn_dilation),
            ),
            self.activate_func,
        )
