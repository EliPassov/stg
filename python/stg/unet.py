"""
Simple Unet Structure.
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


from .layers import FeatureSelector, GatingNet
from .models import mu_return, ModelIOKeysMixin


# Normalization_Dict = {
#     'group': partial(nn.GroupNorm, num)
# }


class Conv3(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )

        self.is_res = is_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main(x)
        if self.is_res:
            x = x + self.conv(x)
            return x / 1.414
        else:
            return self.conv(x)


class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetDown, self).__init__()
        layers = [Conv3(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            Conv3(out_channels, out_channels),
            Conv3(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, skip), 1)
        x = self.model(x)

        return x


# class TimeSiren(nn.Module):
#     def __init__(self, emb_dim: int) -> None:
#         super(TimeSiren, self).__init__()
#
#         self.lin1 = nn.Linear(1, emb_dim, bias=False)
#         self.lin2 = nn.Linear(emb_dim, emb_dim)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.view(-1, 1)
#         x = torch.sin(self.lin1(x))
#         x = self.lin2(x)
#         return x


class NaiveUnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_feat: int, noise_sigma=0, feature_selector: nn.Module = None) -> None:
        super(NaiveUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_feat = n_feat
        self.noise_sigma = noise_sigma

        self.init_conv = Conv3(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())
        self.feature_selector = feature_selector
        # self.timeembed = TimeSiren(2 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),
            # nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out_conv = nn.Conv2d(2 * n_feat, self.out_channels, 3, 1, 1)

    # def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        if self.training:
            noise = torch.randn_like(x)*self.noise_sigma
            x = x+noise

        x = self.init_conv(x)

        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        encoding = self.to_vec(down3)

        if self.feature_selector is not None:
            gated_encoding, mu = self.feature_selector(torch.squeeze(encoding))
            gated_encoding = torch.unsqueeze(torch.unsqueeze(gated_encoding, -1), -1)
        else:
            gated_encoding = encoding
            mu = None

        thro = self.up0(gated_encoding)

        up1 = self.up1(thro, torch.zeros_like(down3))
        up2 = self.up2(up1, torch.zeros_like(down2))
        up3 = self.up3(up2, torch.zeros_like(down1))

        out = self.out_conv(torch.cat((up3, torch.zeros_like(x)), 1))

        return out, mu, gated_encoding


class GatedUnet(nn.Module, ModelIOKeysMixin):
    def __init__(self, in_channels, hidden_dims, gating_net_hidden_dims, device, batch_norm=None, dropout=None,
                 activation='relu', sigma=1.0, lam=0.1, lam_sim=0.0, feature_selection=True, noise_sigma=0.0):
        super().__init__()

        if feature_selection:
            self.FeatureSelector = GatingNet(2*hidden_dims, gating_net_hidden_dims, sigma,
                                             device, activation=activation, batch_norm=batch_norm,
                                             dropout=dropout)
        else:
            self.FeatureSelector = None

        self.encoding_model = NaiveUnet(in_channels=in_channels, out_channels=in_channels, n_feat=hidden_dims,
                                        noise_sigma=noise_sigma, feature_selector=self.FeatureSelector)
        self.lam = lam
        self.lam_sim = lam_sim
        self.loss = nn.MSELoss()
        self.count = 1

    def forward(self, feed_dict):
        x = self._get_input((feed_dict))
        # x, mu = self.FeatureSelector(self._get_input(feed_dict))
        pred, mu, _ = self.encoding_model.forward(x)
        if self.training:
            loss = self.loss(pred, x)
            if self.FeatureSelector is not None:
                reg = torch.mean(self.FeatureSelector.regularizer((mu + 0.5) / self.FeatureSelector.sigma))
                similarity_reg = self.FeatureSelector.similarity_loss(mu)
                total_loss = loss + self.lam * reg + self.lam_sim * similarity_reg
                if self.count % 100 == 0:
                    print(total_loss.item(), loss.item(), reg.item(), similarity_reg.item(), ((mu>0).sum()/x.size(0)).item(), mu.shape)
                self.count += 1
            else:
                total_loss = loss
            return total_loss, dict(), dict()
        else:
            return self._compose_output(pred)

    def freeze_weights(self):
        for name, p in self.named_parameters():
            if name != 'mu':
                p.requires_grad = False

    def get_gates(self, mode, x):
        _, mu, _ = self.encoding_model.forward(x)
        return mu_return(mu, mode)

