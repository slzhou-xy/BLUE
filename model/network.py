import torch
import torch.nn as nn
import torch.nn.functional as F
from model.embedding import PosEmbedding, SpaEmbedding, TemporalEmbedding


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        d_model = args.d_model
        n_heads = args.n_heads
        d_ff = args.d_ff
        dropout = args.dropout
        n_layers_s2 = args.n_layers_s2
        n_layers_s3 = args.n_layers_s3
        n_layers_s5 = args.n_layers_s5

        layer_s2 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )

        self.decoder_s2 = nn.TransformerEncoder(
            encoder_layer=layer_s2,
            num_layers=n_layers_s2
        )

        self.decoder_up_s3_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.decoder_up_s3_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.decoder_s3 = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers_s3
        )

        self.decoder_up_s5_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.decoder_up_s5_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.decoder_s5 = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers_s5
        )

    def forward(self, x_s5, x_s3, x_s2, traj_len_s5, traj_len_s3, traj_len_s2):
        max_len = traj_len_s2.max().item()
        padding_mask_s2 = torch.arange(max_len, device=x_s2.device)[None, :] >= traj_len_s2[:, None]

        max_len = traj_len_s3.max().item()
        padding_mask_s3 = torch.arange(max_len, device=x_s2.device)[None, :] >= traj_len_s3[:, None]

        max_len = traj_len_s5.max().item()
        padding_mask_s5 = torch.arange(max_len, device=x_s2.device)[None, :] >= traj_len_s5[:, None]

        x_s2 = self.decoder_s2(x_s2, src_key_padding_mask=padding_mask_s2)

        x_s3_up = self.decoder_up_s3_cross_attn(query=x_s3, key=x_s2, value=x_s2, key_padding_mask=padding_mask_s2)[0]
        x_s3_up = self.decoder_up_s3_attn(query=x_s3_up, key=x_s3_up, value=x_s3_up, key_padding_mask=padding_mask_s3)[0]
        x_s3 = self.decoder_s3(x_s3_up + x_s3, src_key_padding_mask=padding_mask_s3)

        x_s5_up = self.decoder_up_s5_cross_attn(query=x_s5, key=x_s3, value=x_s3, key_padding_mask=padding_mask_s3)[0]
        x_s5_up = self.decoder_up_s5_attn(query=x_s5_up, key=x_s5_up, value=x_s5_up, key_padding_mask=padding_mask_s5)[0]
        x_s5 = self.decoder_s5(x_s5_up + x_s5, src_key_padding_mask=padding_mask_s5)

        return x_s5


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        d_model = args.d_model
        n_heads = args.n_heads
        d_ff = args.d_ff
        dropout = args.dropout
        n_layers_s5 = args.n_layers_s5
        n_layers_s3 = args.n_layers_s3
        n_layers_s2 = args.n_layers_s2

        self.patchify = args.patchify
        self.max_patch_len_s2 = args.max_patch_len_s2
        self.max_patch_len_s3 = args.max_patch_len_s3
        self.cls = args.cls

        self.cls_token = None
        if self.cls is True:
            self.cls_token = nn.Parameter(torch.randn(d_model))

        self.pe = PosEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

        layer_s5 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder_s5 = nn.TransformerEncoder(
            encoder_layer=layer_s5,
            num_layers=n_layers_s5
        )

        layer_s3 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder_s3 = nn.TransformerEncoder(
            encoder_layer=layer_s3,
            num_layers=n_layers_s3
        )

        layer_s2 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder_s2 = nn.TransformerEncoder(encoder_layer=layer_s2, num_layers=n_layers_s2)

        self.patch_attn_s3 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

        self.patch_attn_s2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, traj_len_s5, traj_len_s3, patch_len_s3, traj_len_s2, patch_len_s2):
        _, _, d_model = x.shape

        # * s5
        if self.cls_token is not None:
            cls_token = self.cls_token.unsqueeze(0).repeat(x.shape[0], 1, 1)
            x = torch.cat([cls_token, x], dim=1)
            traj_len_s5 += 1

        max_len = traj_len_s5.max().item()
        padding_mask = torch.arange(max_len, device=x.device)[None, :] >= traj_len_s5[:, None]

        x = x + self.pe(x)
        x_s5 = self.encoder_s5(x, src_key_padding_mask=padding_mask)

        # * s5 -> s3
        if self.cls is True:
            x_s5_wo_cls = x_s5[:, 1:]
            padding_mask = padding_mask[:, 1:]
            cls_token = x_s5[:, 0]
        else:
            x_s5_wo_cls = x_s5
            cls_token = None

        padding_mask = padding_mask.reshape(-1)
        x_s5_flatten = x_s5_wo_cls.reshape(-1, x_s5_wo_cls.size(-1))
        x_s5_flatten = x_s5_flatten[~padding_mask]

        # 利用 torch.repeat_interleave 进行张量分割的等价操作
        patch_offsets = torch.cumsum(torch.tensor([0] + patch_len_s3.tolist()[:-1], device=x.device), dim=0)
        patch_indices = torch.repeat_interleave(torch.arange(patch_len_s3.shape[0], device=x.device), patch_len_s3)

        x_s3_padded = x.new_zeros(patch_len_s3.shape[0], self.max_patch_len_s3, d_model)
        x_s3_padded[patch_indices, torch.arange(x_s5_flatten.size(0), device=x.device) - patch_offsets[patch_indices]] = x_s5_flatten
        attn_mask = torch.arange(self.max_patch_len_s3, device=x.device)[None, :] >= patch_len_s3[:, None]
        w = self.patch_attn_s3(x_s3_padded)
        w[attn_mask] = float('-inf')
        w = torch.softmax(w, dim=1)
        x_s3 = torch.sum(w * x_s3_padded, dim=1)

        del patch_offsets, patch_indices

        # 填充并对齐 `x_s3`
        traj_cumsum = torch.cumsum(torch.tensor([0] + traj_len_s3[:-1].tolist(), device=x.device), dim=0)
        traj_indices = torch.repeat_interleave(torch.arange(traj_len_s3.shape[0], device=x.device), traj_len_s3)
        max_len = traj_len_s3.max().item()
        x_s3_aligned = x.new_zeros(traj_len_s3.shape[0], traj_len_s3.max().item(), d_model)
        x_s3_aligned[traj_indices, torch.arange(x_s3.shape[0], device=x.device) - traj_cumsum[traj_indices]] = x_s3
        del traj_cumsum, traj_indices

        if cls_token is not None:
            max_len += 1
            traj_len_s3 += 1
            x_s3 = torch.cat([cls_token.unsqueeze(1), x_s3_aligned], dim=1)
        padding_mask = torch.arange(max_len, device=x.device)[None, :] >= traj_len_s3[:, None]
        x_s3 = x_s3 + self.pe(x_s3)
        x_s3 = self.encoder_s3(x_s3, src_key_padding_mask=padding_mask)

        # * s3 -> s2
        if self.cls is True:
            x_s3_wo_cls = x_s3[:, 1:]
            padding_mask = padding_mask[:, 1:]
            cls_token = x_s3[:, 0]
        else:
            x_s3_wo_cls = x_s3
            cls_token = None

        padding_mask = padding_mask.reshape(-1)
        x_s3_flatten = x_s3_wo_cls.reshape(-1, x_s3_wo_cls.size(-1))
        x_s3_flatten = x_s3_flatten[~padding_mask]

        # 利用 torch.repeat_interleave 进行张量分割的等价操作
        patch_offsets = torch.cumsum(torch.tensor([0] + patch_len_s2.tolist()[:-1], device=x.device), dim=0)
        patch_indices = torch.repeat_interleave(torch.arange(patch_len_s2.shape[0], device=x.device), patch_len_s2)

        x_s2_padded = x.new_zeros(patch_len_s2.shape[0],  self.max_patch_len_s2, d_model)
        x_s2_padded[patch_indices, torch.arange(x_s3_flatten.size(0), device=x.device) - patch_offsets[patch_indices]] = x_s3_flatten
        attn_mask = torch.arange(self.max_patch_len_s2, device=x.device)[None, :] >= patch_len_s2[:, None]
        w = self.patch_attn_s2(x_s2_padded)
        w[attn_mask] = float('-inf')
        w = torch.softmax(w, dim=1)
        x_s2 = torch.sum(w * x_s2_padded, dim=1)
        del patch_offsets, patch_indices

        traj_cumsum = torch.cumsum(torch.tensor([0] + traj_len_s2[:-1].tolist(), device=x.device), dim=0)
        traj_indices = torch.repeat_interleave(torch.arange(traj_len_s2.shape[0], device=x.device), traj_len_s2)
        max_len = traj_len_s2.max().item()
        x_s2_aligned = x.new_zeros(traj_len_s2.shape[0], max_len, d_model)
        x_s2_aligned[traj_indices, torch.arange(x_s2.shape[0], device=x.device) - traj_cumsum[traj_indices]] = x_s2
        del traj_cumsum, traj_indices

        if cls_token is not None:
            max_len += 1
            traj_len_s2 += 1
            x_s2 = torch.cat([cls_token.unsqueeze(1), x_s2_aligned], dim=1)
        padding_mask = torch.arange(max_len, device=x.device)[None, :] >= traj_len_s2[:, None]
        assert ((~padding_mask).sum(dim=1) == traj_len_s2).all().item() is True
        x_s2 = x_s2 + self.pe(x_s2)
        x_s2 = self.encoder_s2(x_s2, src_key_padding_mask=padding_mask)

        return x_s5, x_s3, x_s2


class Net(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.cls = args.cls

        self.spa_emb = SpaEmbedding(6, args.d_model)
        self.time_emb = TemporalEmbedding(6, args.d_model)

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.predictor_spa = nn.Sequential(
            nn.Linear(args.d_model, args.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(args.d_model, 6)
        )

        self.predictor_time = nn.Sequential(
            nn.Linear(args.d_model, args.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(args.d_model, 6)
        )

    def forward_encoder(self, data):
        x = data['x']
        time_x = data['time_x']
        traj_len_s5 = data['traj_len_s5']
        patch_len_s2 = data['patch_len_s2']
        traj_len_s2 = data['traj_len_s2']
        patch_len_s3 = data['patch_len_s3']
        traj_len_s3 = data['traj_len_s3']

        x = self.spa_emb(x)
        tx = self.time_emb(time_x)
        x = x + tx

        x_s5, x_s3, x_s2 = self.encoder(x, traj_len_s5, traj_len_s3, patch_len_s3, traj_len_s2, patch_len_s2)

        return x_s2[:, 0]

    def forward_loss(self, data):
        x = data['x']
        mask_y = data['mask_y']
        time_x = data['time_x']
        traj_len_s5 = data['traj_len_s5']
        patch_len_s2 = data['patch_len_s2']
        traj_len_s2 = data['traj_len_s2']
        patch_len_s3 = data['patch_len_s3']
        traj_len_s3 = data['traj_len_s3']

        x = self.spa_emb(x)
        tx = self.time_emb(time_x)
        x = x + tx

        # if cls_token is not None, traj_len_s5 and traj_len_s2 will be increased by 1 in encoder
        x_s5, x_s3, x_s2 = self.encoder(x, traj_len_s5, traj_len_s3, patch_len_s3, traj_len_s2, patch_len_s2)
        x_s5_out = self.decoder(x_s5, x_s3, x_s2, traj_len_s5, traj_len_s3, traj_len_s2)

        # remove cls token
        x_s5_out = x_s5_out[:, 1:]

        mask_y = mask_y.reshape(-1).bool()
        x_s5_out = x_s5_out.reshape(-1, x_s5_out.shape[-1])
        x_s5_out = x_s5_out[mask_y]

        spa_y = data['x']
        spa_y = spa_y.reshape(-1, spa_y.shape[-1])
        spa_y = spa_y[mask_y]
        spa_y_hat = self.predictor_spa(x_s5_out)
        spa_loss = F.mse_loss(spa_y_hat, spa_y)

        time_y = data['time_x']
        time_y = time_y.reshape(-1, time_y.shape[-1])
        time_y = time_y[mask_y]
        time_y_hat = self.predictor_time(x_s5_out)
        time_loss = F.mse_loss(time_y_hat, time_y)

        return spa_loss, time_loss
