import torch
import torch.nn as nn
from einops import rearrange

class Encoder(nn.Module):
    def __init__(self, input_channels: int,
                 latent_dim: int,
                 hidden_channels: list = None,
                 fc_dims: list = None,
                 input_size: tuple = (64, 64),
                 kernel_size=3):
        super().__init__()
        self.latent_dim = latent_dim
        if hidden_channels is None:
            hidden_channels = [32, 32, 32, 32]

        if fc_dims is None:
            fc_dims = [256]*2

        layers = []
        in_channels = input_channels

        for h in hidden_channels:
            layers.append(
                nn.Conv2d(in_channels, h, kernel_size=kernel_size,
                          stride=2, padding=1)
            )
            layers.append(nn.ReLU())
            in_channels = h

        self.conv_net = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        # infer flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, *input_size)
            conv_dummy = self.conv_net(dummy_input)
            conv_out_dim = self.flatten(conv_dummy).shape[1]
            conv_out_shape = conv_dummy.shape[1:]

        self.conv_out_dim = conv_out_dim
        self.conv_out_shape = conv_out_shape
        fc_layers = []
        in_features = self.conv_out_dim
        for dim in fc_dims:
            fc_layers.append(nn.Linear(in_features, dim))
            fc_layers.append(nn.ReLU())
            in_features = dim

        self.fc = nn.Sequential(*fc_layers)

        self.fc_mu = nn.Linear(in_features, self.latent_dim)
        self.fc_logvar = nn.Linear(in_features, self.latent_dim)

    def forward(self, x):
        x = self.conv_net(x)
        x = self.flatten(x)
        x = self.fc(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        output_channels: int,
        latent_dim: int,
        conv_output_shape: tuple,
        conv_out_dim: int,
        fc_dims: list = None,
        hidden_channels: list = None,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
    ):
        super().__init__()

        if fc_dims is None:
            fc_dims = [256, 256]

        if hidden_channels is None:
            hidden_channels = [32, 32, 32, 32]

        self.conv_output_shape = conv_output_shape
        self.conv_out_dim = conv_out_dim

        # Reverse the encoder FC path
        fc_layers = []
        in_features = latent_dim

        for dim in reversed(fc_dims):
            fc_layers.append(nn.Linear(in_features, dim))
            fc_layers.append(nn.ReLU())
            in_features = dim

        # Final FC to reach flattened conv feature map
        fc_layers.append(nn.Linear(in_features, self.conv_out_dim))
        fc_layers.append(nn.ReLU())

        self.fc = nn.Sequential(*fc_layers)

        # Reverse conv path
        deconv_layers = []
        reversed_channels = list(reversed(hidden_channels))

        in_channels = self.conv_output_shape[0]  # e.g. 32

        for i, h in enumerate(reversed_channels[1:]):
            deconv_layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    h,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            deconv_layers.append(nn.ReLU())
            in_channels = h

        # Final layer to image channels
        deconv_layers.append(
            nn.ConvTranspose2d(
                in_channels,
                output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
        )

        self.deconv = nn.Sequential(*deconv_layers)

    def forward(self, z):
        x = self.fc(z)
        x = rearrange(x, 'b (c h w) -> b c h w',
                      c=self.conv_output_shape[0],
                      h=self.conv_output_shape[1],
                      w=self.conv_output_shape[2]
                      )
        x = self.deconv(x)
        # x = torch.sigmoid(x)  # Can end with sigmoid if using BCE loss, or leave as is for MSE loss or BCEWithLogits loss
        return x


class VAE(nn.Module):
    def __init__(self, input_channels=1, latent_dim=10, hidden_channels=None,
                 fc_dims=None, input_size=(64, 64), kernel_size=3, output_padding=1):
        super().__init__()
        self.encoder = Encoder(input_channels=input_channels,
                               latent_dim=latent_dim,
                               hidden_channels=hidden_channels,
                               fc_dims=fc_dims,
                               input_size=input_size,
                               kernel_size=kernel_size)
        self.decoder = Decoder(
            output_channels=input_channels,
            latent_dim=latent_dim,
            conv_output_shape=self.encoder.conv_out_shape,
            conv_out_dim=self.encoder.conv_out_dim,
            fc_dims=fc_dims,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            output_padding=output_padding
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
