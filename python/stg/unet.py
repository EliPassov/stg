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
    def __init__(self, in_channels: int, out_channels: int, feature_selector: nn.Module = None) -> None:
        super(UnetDown, self).__init__()
        layers = [Conv3(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)
        self.feature_selector = feature_selector

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x_out = self.model(x)
        if self.feature_selector is not None:
            x_out, mu, stochastic_gate, losses = self.feature_selector(x_out, include_losses=self.training)
        else:
            stochastic_gate = None
            losses = None
            mu = None
        return x_out, mu, stochastic_gate, losses


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
    def __init__(self, in_channels: int, out_channels: int, n_feat: int, noise_sigma=0,
                 feature_selector: nn.Module = None, include_skip_connection: bool = False) -> None:
        super(NaiveUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_feat = n_feat
        self.noise_sigma = noise_sigma
        self.include_skip_connection = include_skip_connection

        self.init_conv = Conv3(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat, feature_selector[0] if feature_selector is not None and isinstance(feature_selector, list) else None)
        self.down2 = UnetDown(n_feat, 2 * n_feat, feature_selector[1] if feature_selector is not None and isinstance(feature_selector, list) else None)
        self.down3 = UnetDown(2 * n_feat, 2 * n_feat, feature_selector[2] if feature_selector is not None and isinstance(feature_selector, list) else None)

        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())
        self.feature_selector = None
        if feature_selector is not None:
            self.feature_selector = feature_selector[-1] if isinstance(feature_selector, list) else feature_selector
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
            x = x + noise

        x = self.init_conv(x)

        down1, _, gates1, losses1 = self.down1(x)
        down2, _, gates2, losses2 = self.down2(down1)
        down3, _, gates3, losses3 = self.down3(down2)

        encoding = self.to_vec(down3)

        if self.feature_selector is not None:
            gated_encoding, mu, stochastic_gate, losses = self.feature_selector(torch.squeeze(encoding), include_losses=self.training)
            gated_encoding = torch.unsqueeze(torch.unsqueeze(gated_encoding, -1), -1)
        else:
            gated_encoding = encoding
            losses = None
            mu = None

        thro = self.up0(gated_encoding)

        if gates3 is not None:
            thro = thro * gates3
        up1 = self.up1(thro, down3 if self.include_skip_connection else torch.zeros_like(down3))
        if gates2 is not None:
            up1 = up1 * gates2
        up2 = self.up2(up1, down2 if self.include_skip_connection else torch.zeros_like(down2))
        if gates1 is not None:
            up2 = up2 * gates1
        up3 = self.up3(up2, down1 if self.include_skip_connection else torch.zeros_like(down1))

        out = self.out_conv(torch.cat((up3, torch.zeros_like(x)), 1))

        if losses1 is not None:
            losses = ([losses[0]], [losses[1]])
            for lo in [losses1, losses2, losses3]:
                losses[0].append(lo[0])
                losses[1].append(lo[1])
            losses = (torch.stack(losses[0], 0), torch.stack(losses[1], 0))
        return out, losses, mu, gated_encoding


class GatedUnet(nn.Module, ModelIOKeysMixin):
    def __init__(self, in_channels, hidden_dims, gating_net_hidden_dims, device, batch_norm=None, dropout=None,
                 activation='relu', sigma=1.0, lam=0.1, lam_sim=0.0, feature_selection="latent", noise_sigma=0.0,
                 include_skip_connection=False):
        super().__init__()

        if feature_selection == "latent" or feature_selection is True:
            self.FeatureSelector = GatingNet(2*hidden_dims, gating_net_hidden_dims, sigma, device,
                                             activation=activation, batch_norm=batch_norm, dropout=dropout)
        elif feature_selection == 'all':
            self.FeatureSelector = []
            if isinstance(lam, float):
                lam = torch.ones(4, device=device) * lam
            else:
                lam = torch.Tensor(lam).to(device)
            if isinstance(lam_sim, float):
                lam_sim = torch.ones(4, device=device) * lam_sim
            else:
                lam_sim = torch.Tensor(lam_sim).to(device)
            half_gating_net_hidden_dims = [g // 2 for g in gating_net_hidden_dims] if isinstance(gating_net_hidden_dims, list) else gating_net_hidden_dims // 2
            self.FeatureSelector.append(GatingNet(hidden_dims, half_gating_net_hidden_dims, sigma, device,
                                                  activation=activation, batch_norm=batch_norm, dropout=dropout,
                                                  pooling=True))
            self.FeatureSelector.append(GatingNet(2*hidden_dims, gating_net_hidden_dims, sigma, device,
                                                  activation=activation, batch_norm=batch_norm, dropout=dropout,
                                                  pooling=True))
            self.FeatureSelector.append(GatingNet(2*hidden_dims, gating_net_hidden_dims, sigma, device,
                                                  activation=activation, batch_norm=batch_norm, dropout=dropout,
                                                  pooling=True))
            self.FeatureSelector.append(GatingNet(2*hidden_dims, gating_net_hidden_dims, sigma, device,
                                                  activation=activation, batch_norm=batch_norm, dropout=dropout))
        elif feature_selection == 'none':
            self.FeatureSelector = None
        else:
            raise ValueError('Unsupported value for features_selection ', feature_selection)

        self.encoding_model = NaiveUnet(in_channels=in_channels, out_channels=in_channels, n_feat=hidden_dims,
                                        noise_sigma=noise_sigma, feature_selector=self.FeatureSelector,
                                        include_skip_connection=include_skip_connection)
        self.lam = lam
        self.lam_sim = lam_sim
        self.loss = nn.MSELoss()
        self.count = 1

    def forward(self, feed_dict):
        x = self._get_input((feed_dict))
        # x, mu = self.FeatureSelector(self._get_input(feed_dict))
        pred, losses, mu, _ = self.encoding_model.forward(x)
        if self.training:
            loss = self.loss(pred, x)
            if self.FeatureSelector is not None:
                # reg = torch.mean(self.FeatureSelector.regularizer((mu + 0.5) / self.FeatureSelector.sigma))
                # similarity_reg = self.FeatureSelector.similarity_loss(mu)
                reg, similarity_loss = losses
                total_loss = loss + (self.lam * reg).sum() + (self.lam_sim * similarity_loss).sum()
                if self.count % 100 == 0:
                    if isinstance(self.FeatureSelector, list):
                        print(total_loss.item(), loss.item(), reg[0].item(), reg[-1].item(), similarity_loss[0].item(), similarity_loss[-1].item(), ((mu>0).sum()/x.size(0)).item())
                    else:
                        print(total_loss.item(), loss.item(), reg.item(), similarity_loss.item(), ((mu>0).sum()/x.size(0)).item())
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
        _, _, mu, _ = self.encoding_model.forward(x)
        return mu_return(mu, mode)

