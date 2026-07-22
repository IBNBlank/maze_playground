# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""DETRVAE: optional CVAE encoder + DETR action-chunk decoder."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from utils.policy.helper.detr.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
)


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + std * eps


def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    def get_position_angle_vec(position: int):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """Query-token action chunker with optional CVAE latent.

    - ``encoder is not None`` (ACT): train samples posterior ``z``; eval
      samples prior ``z ~ N(0, I)``.
    - ``encoder is None`` (BC): always ``z=0``; same decoder / ``Tanh`` head.
    """

    def __init__(
        self,
        transformer,
        encoder,
        state_dim,
        action_dim,
        num_queries,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)

        self.latent_dim = 32
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

        if encoder is not None:
            self.cls_embed = nn.Embedding(1, hidden_dim)
            self.encoder_state_proj = nn.Linear(state_dim, hidden_dim)
            self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
            self.register_buffer(
                "pos_table",
                get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim),
            )

    def _latent_from_prior(
        self,
        bs: int,
        device: torch.device,
        *,
        sample: bool,
    ) -> torch.Tensor:
        if sample:
            latent_sample = torch.randn(
                bs, self.latent_dim, dtype=torch.float32, device=device)
        else:
            latent_sample = torch.zeros(
                bs, self.latent_dim, dtype=torch.float32, device=device)
        return self.latent_out_proj(latent_sample)

    def forward(self, state: torch.Tensor, actions: torch.Tensor | None = None):
        bs = state.shape[0]
        use_cvae = self.encoder is not None and actions is not None

        if use_cvae:
            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            state_embed = self.encoder_state_proj(state).unsqueeze(1)
            action_embed = self.encoder_action_proj(actions)
            encoder_input = torch.cat(
                [cls_embed, state_embed, action_embed], dim=1).permute(1, 0, 2)
            is_pad = torch.full(
                (bs, encoder_input.shape[0]),
                False,
                device=state.device,
            )
            pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)
            encoder_output = self.encoder(
                encoder_input,
                pos=pos_embed,
                src_key_padding_mask=is_pad,
            )
            latent_info = self.latent_proj(encoder_output[0])
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_input = self.latent_out_proj(reparametrize(mu, logvar))
        else:
            mu = logvar = None
            # ACT eval: sample prior; BC: deterministic z=0
            latent_input = self._latent_from_prior(
                bs, state.device, sample=self.encoder is not None)

        state_feat = self.input_proj_robot_state(state)
        hs = self.transformer(
            None,
            None,
            self.query_embed.weight,
            None,
            latent_input,
            state_feat,
            self.additional_pos_embed.weight,
        )[0]
        a_hat = self.action_head(hs)
        return a_hat, [mu, logvar]


def build_encoder(args):
    d_model = args.hidden_dim
    encoder_layer = TransformerEncoderLayer(
        d_model,
        args.nheads,
        args.dim_feedforward,
        args.dropout,
        "relu",
        args.pre_norm,
    )
    encoder_norm = nn.LayerNorm(d_model) if args.pre_norm else None
    return TransformerEncoder(encoder_layer, args.enc_layers, encoder_norm)
