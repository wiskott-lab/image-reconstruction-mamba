import torch
import torch.nn as nn
from timm.layers import DropPath
import config


class TransformerBaseline(nn.Module):

    def __init__(self, embed_dim, foveation_properties, num_heads=8, num_layers=6, ff_dim=4, num_channels=3,drop_path=0.05,
                 pos_emb_type='add', decoder=None):
        super(TransformerBaseline, self).__init__()
        self.pos_emb_type = pos_emb_type
        self.foveation_properties = foveation_properties
        self.backbone = ConvBackbone(kernel_size=foveation_properties.patch_sizes[0], in_channels=num_channels, out_features=embed_dim)
        self.encoder = Encoder(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,  ff_dim=ff_dim,
                             drop_path=drop_path)
        self.decoder = Decoder(in_features=embed_dim, out_channels=num_channels) if decoder is None else decoder
        self.reg = nn.Parameter(torch.rand(size=(1, embed_dim)) * .02)



    def forward_seq(self, seq, pos_x, pos_y, *args, **kwargs):
        seq = self.backbone(seq[0])
        seq = self.pos_embed(seq, pos_x, pos_y)
        reg = self.reg.unsqueeze(0).expand(seq.shape[0], -1, -1)
        seq = torch.concat([reg, seq], dim=1)
        seq = self.encoder(seq)
        outputs = self.decoder(seq[:, 0])
        return outputs

    def pos_embed(self, token, pos_x, pos_y):
        if self.pos_emb_type == 'concat':
            if len(token.shape) == 3:
                token =  torch.cat([token, pos_x.unsqueeze(2) / 100, pos_y.unsqueeze(2) / 100], dim=2)
            else:
                token = torch.cat([token, pos_x.unsqueeze(1) / 100, pos_y.unsqueeze(1) / 100], dim=1)  # token += get_pos_emb(pos_x=self.pos_x, pos_y=self.pos_y, dim=token.shape[-1])
        elif self.pos_emb_type == 'add':
            pos_emb = get_pos_emb(pos_x, pos_y, dim=token.shape[-1])
            token = token + pos_emb
        return token


class TransformerSequential(nn.Module):

    def __init__(self, embed_dim, foveation_properties, num_heads=8, num_layers=6, ff_dim=4, num_channels=3,drop_path=0.05,
                 pos_emb_type='add', decoder=None):
        super(TransformerSequential, self).__init__()
        self.pos_emb_type = pos_emb_type
        self.foveation_properties = foveation_properties
        self.backbone = ConvBackbone(kernel_size=foveation_properties.patch_sizes[0], in_channels=num_channels, out_features=embed_dim)
        self.encoder = Encoder(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,  ff_dim=ff_dim,
                             drop_path=drop_path)
        self.decoder = Decoder(in_features=embed_dim, out_channels=num_channels) if decoder is None else decoder
        self.query_token = nn.Parameter(torch.randn(size=(embed_dim,)) * .02)



    def forward_seq(self, seq, pos_x, pos_y, query_pos_x, query_pos_y):
        seq = self.backbone(seq[0])
        img_seq = self.pos_embed(seq, pos_x, pos_y)
        query_token = self.query_token.expand(seq.shape[0], query_pos_x.shape[1], -1)
        query_token = self.pos_embed(query_token, query_pos_x, query_pos_y)

        seq = torch.concat([seq, query_token], dim=1)
        seq = self.encoder(seq)
        outputs = self.decoder(seq[:, img_seq.shape[1]:])
        return outputs

    def pos_embed(self, token, pos_x, pos_y):
        if self.pos_emb_type == 'concat':
            if len(token.shape) == 3:
                token =  torch.cat([token, pos_x.unsqueeze(2) / 100, pos_y.unsqueeze(2) / 100], dim=2)
            else:
                token = torch.cat([token, pos_x.unsqueeze(1) / 100, pos_y.unsqueeze(1) / 100], dim=1)  # token += get_pos_emb(pos_x=self.pos_x, pos_y=self.pos_y, dim=token.shape[-1])
        elif self.pos_emb_type == 'add':
            pos_emb = get_pos_emb(pos_x, pos_y, dim=token.shape[-1])
            token = token + pos_emb
        return token

class TransformerEncoderDecoder(nn.Module):

    def __init__(self, embed_dim, foveation_properties, num_heads=8, num_layers=6, ff_dim=4, num_channels=3,drop_path=0.05,
                 pos_emb_type='add', decoder=None):
        super(TransformerEncoderDecoder, self).__init__()
        self.pos_emb_type = pos_emb_type
        self.foveation_properties = foveation_properties
        self.backbone = ConvBackbone(kernel_size=foveation_properties.patch_sizes[0], in_channels=num_channels, out_features=embed_dim)
        self.encoder = Encoder(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers // 2,  ff_dim=ff_dim,
                             drop_path=drop_path)
        self.transformer_decoder = TransformerDecoder(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers // 2,  ff_dim=ff_dim,
                             drop_path=drop_path)
        self.head = Decoder(in_features=embed_dim, out_channels=num_channels) if decoder is None else decoder
        self.query_token = nn.Parameter(torch.randn(size=(embed_dim,)) * .02)


    def forward_seq(self, seq, pos_x, pos_y, query_pos_x, query_pos_y):
        seq = self.backbone(seq[0])
        img_seq = self.pos_embed(seq, pos_x, pos_y)
        encoder_seq = self.encoder(img_seq)

        query_token = self.query_token.expand(seq.shape[0], query_pos_x.shape[1], -1)
        decoder_seq = self.pos_embed(query_token, query_pos_x, query_pos_y)

        decoder_out = self.transformer_decoder(decoder_seq=decoder_seq, encoder_seq=encoder_seq)
        outputs = self.head(decoder_out)
        return outputs

    def pos_embed(self, token, pos_x, pos_y):
        if self.pos_emb_type == 'concat':
            if len(token.shape) == 3:
                token =  torch.cat([token, pos_x.unsqueeze(2) / 100, pos_y.unsqueeze(2) / 100], dim=2)
            else:
                token = torch.cat([token, pos_x.unsqueeze(1) / 100, pos_y.unsqueeze(1) / 100], dim=1)  # token += get_pos_emb(pos_x=self.pos_x, pos_y=self.pos_y, dim=token.shape[-1])
        elif self.pos_emb_type == 'add':
            pos_emb = get_pos_emb(pos_x, pos_y, dim=token.shape[-1])
            token = token + pos_emb
        return token

class PatchDecoder(nn.Module):
    def __init__(self, in_features=784, patch_size=5, out_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.fc = nn.Linear(in_features, out_channels * (patch_size**2))

    def forward(self, x, *args, **kwargs):
        B, L = x.shape[0], x.shape[1]
        x = self.fc(x)
        x = torch.sigmoid(x)
        x = x.view(B, L, self.out_channels, self.patch_size, self.patch_size)
        return x



class ConvBackbone(nn.Module):
    def __init__(self, in_channels=3, kernel_size=5, out_features=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_features, kernel_size=kernel_size, padding=0)


    def forward(self, x):
        B, L = x.shape[0], x.shape[1]
        x = x.reshape(B * L, *x.shape[2:])
        x = self.proj(x)
        x = x.reshape(B, L, x.shape[1])
        return x


class MLPBlock(nn.Module):

    def __init__(self, embed_dim, ff_dim, drop_path, layer_scale, dropout):
        super(MLPBlock, self).__init__()
        self.drop_path_mlp = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_scale = nn.Parameter(layer_scale * torch.ones(embed_dim))
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * ff_dim), nn.GELU(),
                                 nn.Linear(embed_dim * ff_dim, embed_dim),
                                 nn.Dropout(dropout))

    def forward(self, seq):
        seq = seq + self.drop_path_mlp(self.layer_scale * self.mlp(self.norm(seq)))
        return seq


class SelfAttnBlock(nn.Module):

    def __init__(self, embed_dim, dropout, drop_path, num_heads, layer_scale):
        super(SelfAttnBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_scale = nn.Parameter(layer_scale * torch.ones(embed_dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, seq):
        seq_normed = self.norm(seq)
        attn, _ = self.attn(query=seq_normed, key=seq_normed, value=seq_normed)
        seq = seq + self.drop_path(self.layer_scale * attn)
        return seq

class CrossAttnBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, drop_path, layer_scale):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.layer_scale = nn.Parameter(layer_scale * torch.ones(embed_dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, decoder_seq, encoder_seq, attn_mask=None):
        seq_normed = self.norm(decoder_seq)
        attn, _ = self.attn(query=seq_normed, key=encoder_seq, value=encoder_seq, attn_mask=attn_mask)
        decoder_seq = decoder_seq + self.drop_path(self.layer_scale * attn)
        return decoder_seq



class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout, drop_path, layer_scale):
        super(DecoderBlock, self).__init__()

        self.self_attn_block = SelfAttnBlock(embed_dim=embed_dim, dropout=dropout, num_heads=num_heads,
                                        drop_path=drop_path, layer_scale=layer_scale)

        self.cross_attn_block = CrossAttnBlock(embed_dim=embed_dim, dropout=dropout, num_heads=num_heads,
                                               drop_path=drop_path,layer_scale=layer_scale)
        self.mlp_block = MLPBlock(embed_dim=embed_dim, ff_dim=ff_dim, drop_path=drop_path,
                                  dropout=dropout, layer_scale=layer_scale)

    def forward(self, decoder_seq, encoder_seq):
        decoder_seq = self.self_attn_block(seq=decoder_seq)
        decoder_seq = self.cross_attn_block(decoder_seq=decoder_seq, encoder_seq=encoder_seq)
        decoder_seq = self.mlp_block(decoder_seq)
        return decoder_seq

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout, drop_path, layer_scale):
        super(EncoderBlock, self).__init__()

        self.attn_block = SelfAttnBlock(embed_dim=embed_dim, dropout=dropout, num_heads=num_heads,
                                        drop_path=drop_path, layer_scale=layer_scale)
        self.mlp_block = MLPBlock(embed_dim=embed_dim, ff_dim=ff_dim, drop_path=drop_path,
                                  dropout=dropout, layer_scale=layer_scale)

    def forward(self, seq):
        seq = self.attn_block(seq=seq)
        seq = self.mlp_block(seq=seq)
        return seq

class Encoder(nn.Module):

    def __init__(self, embed_dim=32, num_heads=8, num_layers=6, ff_dim=4,dropout=0.0, drop_path=0.2, layer_scale=1e-4):
        super(Encoder, self).__init__()
        dpr = [drop_path for _ in range(num_layers)]
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout, drop_path=dpr[i],
                     layer_scale=layer_scale)
            for i in range(num_layers)
        ])

    def forward(self, seq):
        for layer in self.layers:
            seq = layer(seq)
        return seq


class TransformerDecoder(nn.Module):

    def __init__(self, embed_dim=32, num_heads=8, num_layers=6, ff_dim=4,dropout=0.0, drop_path=0.2, layer_scale=1e-4):
        super(TransformerDecoder, self).__init__()
        dpr = [drop_path for _ in range(num_layers)]
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout, drop_path=dpr[i],
                     layer_scale=layer_scale)
            for i in range(num_layers)
        ])

    def forward(self, decoder_seq, encoder_seq):
        for layer in self.layers:
            decoder_seq = layer(decoder_seq=decoder_seq, encoder_seq=encoder_seq)
        return decoder_seq

class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, width=None, norm='rms'):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),  # x2 spatial
            nn.RMSNorm([out_ch, width, width]) if norm == 'rms' else nn.BatchNorm2d(out_ch) ,
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.RMSNorm([out_ch, width, width]) if norm == 'rms' else nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class LinearDecoder(nn.Module):
    def __init__(self, in_features=784, output_shape=(1, 128, 128)):
        super().__init__()
        self.fc = nn.Linear(in_features, output_shape[0] * output_shape[1] * output_shape[2])
        self.output_shape = output_shape
        # Final RGB projection

    def forward(self, x, *args, **kwargs):
        x = self.fc(x)  # (B, base_ch*4*4)
        x = torch.sigmoid(x)
        x = x.reshape(-1, *self.output_shape)
        return x

class Decoder(nn.Module):
    def __init__(self, in_features=784, base_ch=512, norm='batch', out_channels=3):
        super().__init__()
        self.fc = nn.Linear(in_features, base_ch * 4 * 4)
        self.up1 = DeconvBlock(base_ch, base_ch // 2, norm=norm, width=8)  # 4 -> 8
        self.up2 = DeconvBlock(base_ch // 2, base_ch // 4, norm=norm, width=16)  # 8 -> 16
        self.up3 = DeconvBlock(base_ch // 4, base_ch // 8, norm=norm, width=32)  # 16 -> 32
        self.up4 = DeconvBlock(base_ch // 8, base_ch // 16, norm=norm, width=64)  # 32 -> 64
        self.up5 = DeconvBlock(base_ch // 16, base_ch // 32, norm=norm, width=128)  # 64 -> 128

        # Final RGB projection
        self.to_rgb = nn.Conv2d(base_ch // 32, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x, *args, **kwargs):
        x = self.fc(x)  # (B, base_ch*4*4)
        x = x.view(x.shape[0], -1, 4, 4)
        x = self.up1(x)  # (B, base_ch/2, 8, 8)
        x = self.up2(x)  # (B, base_ch/4, 16, 16)
        x = self.up3(x)  # (B, base_ch/8, 32, 32)
        x = self.up4(x)  # (B, base_ch/8, 32, 32)
        x = self.up5(x)  # (B, base_ch/8, 32, 32)

        x = torch.sigmoid(self.to_rgb(x))  # (B, 3, 32, 32)
        return x

class LargeDecoder(nn.Module):
    def __init__(self, in_features=784, base_ch=512, norm='batch'):
        super().__init__()
        self.fc = nn.Linear(in_features, base_ch * 4 * 4)
        self.up1 = DeconvBlock(base_ch, base_ch // 2, norm=norm, width=8)  # 4 -> 8
        self.up2 = DeconvBlock(base_ch // 2, base_ch // 4, norm=norm, width=16)  # 8 -> 16
        self.up3 = DeconvBlock(base_ch // 4, base_ch // 8, norm=norm, width=32)  # 16 -> 32
        self.up4 = DeconvBlock(base_ch // 8, base_ch // 16, norm=norm, width=64)  # 32 -> 64
        self.up5 = DeconvBlock(base_ch // 16, base_ch // 32, norm=norm, width=128)  # 64 -> 128
        self.up6 = DeconvBlock(base_ch // 32, base_ch // 64, norm=norm, width=256)  # 128 -> 128
        # Final RGB projection
        self.to_rgb = nn.Conv2d(base_ch // 64, 3, kernel_size=1, padding=0)

    def forward(self, x, *args, **kwargs):
        x = self.fc(x)  # (B, base_ch*4*4)
        x = x.view(x.shape[0], -1, 4, 4)
        x = self.up1(x)  # (B, base_ch/2, 8, 8)
        x = self.up2(x)  # (B, base_ch/4, 16, 16)
        x = self.up3(x)  # (B, base_ch/8, 32, 32)
        x = self.up4(x)  # (B, base_ch/8, 32, 32)
        x = self.up5(x)  # (B, base_ch/8, 32, 32)
        x = self.up6(x)  # (B, base_ch/8, 32, 32)
        x = torch.sigmoid(self.to_rgb(x))  # (B, 3, 32, 32)
        return x


def get_pos_emb(pos_x, pos_y, dim, temperature=10000):
    omega = torch.arange(dim // 4, dtype=torch.float32, device=config.DEVICE)
    omega = 1. / (temperature ** (omega / (dim / 4)))
    out_x = pos_x[..., None] * omega[None, :]
    sin_x, cos_x = torch.sin(out_x), torch.cos(out_x)
    out_y = pos_y[..., None] * omega[None, :]
    sin_y, cos_y = torch.sin(out_y), torch.cos(out_y)
    pos_emb = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)
    return pos_emb



if __name__ == '__main__':
    decoder = LinearDecoder(in_features=98, output_shape=(1 , 128, 128))
    # model = TransformerBaseline(embed_dim=392, num_heads=8, num_layers=6, ff_dim=4, num_channels=3, drop_path=0.05, pos_emb_type='concat', foveation_properties=)
    a = torch.rand(size=(10, 98))
    decoder(a)
    pass
