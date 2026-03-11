import torch
import torch.nn as nn


class SimpleHead(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features)

    def forward(self, x, *args, **kwargs):
        return self.linear_1(x)


class ReconHead(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, output_shape):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.relu_1 = nn.ReLU()

        self.linear_2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.sigmoid_2 = nn.Sigmoid()
        self.unflatten_2 = nn.Unflatten(dim=-1, unflattened_size=output_shape)

    def forward(self, x, *args, **kwargs):
        x = self.relu_1(self.linear_1(x))
        x = self.unflatten_2(self.sigmoid_2(self.linear_2(x)))
        return x


class UpConvReconHead(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=4096)
        self.relu_1 = nn.ReLU()

        self.up_conv_2 = nn.ConvTranspose2d(4096, 2048, kernel_size=2, stride=1, padding=0, output_padding=0)
        self.bn_2 =  nn.BatchNorm2d(2048)

        self.relu_2 = nn.ReLU() # 2048x2x2

        self.up_conv_3 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=1, padding=0, output_padding=0)
        self.bn_3 =  nn.BatchNorm2d(1024)
        self.relu_3 = nn.ReLU() # 1024x4x4

        self.up_conv_4 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_4 =  nn.BatchNorm2d(512)
        self.relu_4 = nn.ReLU()  # 512x8x8
        #
        self.up_conv_5 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_5 =  nn.BatchNorm2d(256)
        self.relu_5 = nn.ReLU()  # 256x16x16
        #
        self.up_conv_6 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_6 =  nn.BatchNorm2d(128)
        self.relu_6 = nn.ReLU()  # 128x32x32

        self.up_conv_7 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_7 =  nn.BatchNorm2d(64)
        self.relu_7 = nn.ReLU()  # 64x64x64

        self.up_conv_8 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.sig_8 = nn.Sigmoid()  # 512x32x32


    def forward(self, x,  *args, **kwargs):
        B, L = x.shape[0], x.shape[1]
        x = self.relu_1(self.linear_1(x))
        x = x.view(B * L, x.shape[2], 1, 1)
        x = self.relu_2(self.bn_2(self.up_conv_2(x)))
        x = self.relu_3(self.bn_3(self.up_conv_3(x)))
        x = self.relu_4(self.bn_4(self.up_conv_4(x)))
        x = self.relu_5(self.bn_5(self.up_conv_5(x)))
        x = self.relu_6(self.bn_6(self.up_conv_6(x)))
        x = self.relu_7(self.bn_7(self.up_conv_7(x)))
        x = self.sig_8(self.up_conv_8(x))
        x = x.view(B, L, *x.shape[1:])
        return x


class UpConvReconHeadGN(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=4096)
        self.relu_1 = nn.ReLU()

        self.up_conv_2 = nn.ConvTranspose2d(4096, 2048, kernel_size=2, stride=1, padding=0, output_padding=0)
        self.gn_2 =  nn.GroupNorm(8, 2048)
        self.relu_2 = nn.ReLU() # 2048x2x2

        self.up_conv_3 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=1, padding=0, output_padding=0)
        self.gn_3 =  nn.GroupNorm(8, 1024)
        self.relu_3 = nn.ReLU() # 1024x4x4

        self.up_conv_4 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.gn_4 =  nn.GroupNorm(8, 512)
        self.relu_4 = nn.ReLU()  # 512x8x8
        #
        self.up_conv_5 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.gn_5 =  nn.GroupNorm(8, 256)
        self.relu_5 = nn.ReLU()  # 256x16x16
        #
        self.up_conv_6 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.gn_6 =  nn.GroupNorm(8, 128)
        self.relu_6 = nn.ReLU()  # 128x32x32

        self.up_conv_7 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.gn_7 =  nn.GroupNorm(8, 64)
        self.relu_7 = nn.ReLU()  # 64x64x64

        self.up_conv_8 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.sig_8 = nn.Sigmoid()  # 512x32x32


    def forward(self, x,  *args, **kwargs):
        B, L = x.shape[0], x.shape[1]
        x = self.relu_1(self.linear_1(x))
        x = x.view(B * L, x.shape[2], 1, 1)
        x = self.relu_2(self.gn_2(self.up_conv_2(x)))
        x = self.relu_3(self.gn_3(self.up_conv_3(x)))
        x = self.relu_4(self.gn_4(self.up_conv_4(x)))
        x = self.relu_5(self.gn_5(self.up_conv_5(x)))
        x = self.relu_6(self.gn_6(self.up_conv_6(x)))
        x = self.relu_7(self.gn_7(self.up_conv_7(x)))
        x = self.sig_8(self.up_conv_8(x))
        x = x.view(B, L, *x.shape[1:])
        return x


class UpConvReconCifarHead(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=1024)
        self.relu_1 = nn.ReLU()

        self.up_conv_2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_2 = nn.ReLU()  # 512x2x2
        #
        self.up_conv_3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_3 = nn.ReLU()  # 256x4x4
        #
        self.up_conv_4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_4 = nn.ReLU()  # 128x8x8

        self.up_conv_5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_5 = nn.ReLU()  # 64x16x16

        self.up_conv_6 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.sig_6 = nn.Sigmoid()  # 3x32x32


    def forward(self, x,  *args, **kwargs):
        B, L = x.shape[0], x.shape[1]
        x = self.relu_1(self.linear_1(x))
        x = x.view(B * L, x.shape[2], 1, 1)
        x = self.relu_2(self.up_conv_2(x))
        x = self.relu_3(self.up_conv_3(x))
        x = self.relu_4(self.up_conv_4(x))
        x = self.relu_5(self.up_conv_5(x))
        x = self.sig_6(self.up_conv_6(x))
        x = x.view(B, L, *x.shape[1:])
        return x


class UpConvReconCifarHeadBN(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=1024)
        self.relu_1 = nn.ReLU()

        self.up_conv_2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_2 =  nn.BatchNorm2d(512)
        self.relu_2 = nn.ReLU()  # 512x2x2
        #
        self.up_conv_3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_3 =  nn.BatchNorm2d(256)
        self.relu_3 = nn.ReLU()  # 256x4x4
        #
        self.up_conv_4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_4 =  nn.BatchNorm2d(128)
        self.relu_4 = nn.ReLU()  # 128x8x8

        self.up_conv_5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_5 =  nn.BatchNorm2d(64)
        self.relu_5 = nn.ReLU()  # 64x16x16

        self.up_conv_6 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.sig_6 = nn.Sigmoid()  # 3x32x32


    def forward(self, x,  *args, **kwargs):
        B, L = x.shape[0], x.shape[1]
        x = self.relu_1(self.linear_1(x))
        x = x.view(B * L, x.shape[2], 1, 1)
        x = self.relu_2(self.bn_2(self.up_conv_2(x)))
        x = self.relu_3(self.bn_3(self.up_conv_3(x)))
        x = self.relu_4(self.bn_4(self.up_conv_4(x)))
        x = self.relu_5(self.bn_5(self.up_conv_5(x)))
        x = self.sig_6(self.up_conv_6(x))
        x = x.view(B, L, *x.shape[1:])
        return x


class UpConvReconCifarHeadGN(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=1024)
        self.relu_1 = nn.ReLU()

        self.up_conv_2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.gn_2 =  nn.GroupNorm(8, 512)
        self.relu_2 = nn.ReLU()  # 512x2x2
        #
        self.up_conv_3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.gn_3 =  nn.GroupNorm(8, 256)
        self.relu_3 = nn.ReLU()  # 256x4x4
        #
        self.up_conv_4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.gn_4 = nn.GroupNorm(8, 128)
        self.relu_4 = nn.ReLU()  # 128x8x8

        self.up_conv_5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.gn_5 =  nn.GroupNorm(8, 64)
        self.relu_5 = nn.ReLU()  # 64x16x16

        self.up_conv_6 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.sig_6 = nn.Sigmoid()  # 3x32x32


    def forward(self, x,  *args, **kwargs):
        B, L = x.shape[0], x.shape[1]
        x = self.relu_1(self.linear_1(x))
        x = x.view(B * L, x.shape[2], 1, 1)
        x = self.relu_2(self.gn_2(self.up_conv_2(x)))
        x = self.relu_3(self.gn_3(self.up_conv_3(x)))
        x = self.relu_4(self.gn_4(self.up_conv_4(x)))
        x = self.relu_5(self.gn_5(self.up_conv_5(x)))
        x = self.sig_6(self.up_conv_6(x))
        x = x.view(B, L, *x.shape[1:])
        return x


class UpConvReconCifarHeadLarge(nn.Module):


    def __init__(self, in_features):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=2048)
        self.relu_1 = nn.ReLU()

        self.linear_2 = nn.Linear(in_features=2048, out_features=1024)
        self.relu_2 = nn.ReLU()

        self.up_conv_3 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_3 = nn.ReLU()  # 512x2x2
        #
        self.up_conv_4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_4 = nn.ReLU()  # 256x4x4
        #
        self.up_conv_5 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_5 = nn.ReLU()  # 128x8x8

        self.up_conv_6 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_6 = nn.ReLU()  # 64x16x16

        self.up_conv_7 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.sig_7 = nn.Sigmoid()  # 3x32x32


    def forward(self, x,  *args, **kwargs):
        B, L = x.shape[0], x.shape[1]
        x = self.relu_1(self.linear_1(x))
        x = self.relu_2(self.linear_2(x))

        x = x.view(B * L, x.shape[2], 1, 1)
        x = self.relu_3(self.up_conv_3(x))
        x = self.relu_4(self.up_conv_4(x))
        x = self.relu_5(self.up_conv_5(x))
        x = self.relu_6(self.up_conv_6(x))
        x = self.sig_7(self.up_conv_7(x))
        x = x.view(B, L, *x.shape[1:])
        return x




class PosReconHead(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, output_shape):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features+2, out_features=hidden_features)
        self.relu_1 = nn.ReLU()

        self.linear_2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.sigmoid_2 = nn.Sigmoid()
        self.unflatten_2 = nn.Unflatten(dim=-1, unflattened_size=output_shape)

    def forward(self, x, pos_x, pos_y):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        pos_x = (pos_x / 1000)[None, None, :, None].expand(x.shape[0], x.shape[1], -1, 1)
        pos_y = (pos_y / 1000)[None, None, :, None].expand(x.shape[0], x.shape[1], -1, 1)
        x = x[:, :, None, :].expand(x.shape[0], x.shape[1], pos_x.shape[-2], x.shape[2])
        x = torch.cat([x, pos_x, pos_y], dim=-1)
        x = self.relu_1(self.linear_1(x))
        x = self.unflatten_2(self.sigmoid_2(self.linear_2(x)))
        return x
#




class CIFAR32DeconvDecoder(nn.Module):
    def __init__(self, in_features=784, base_ch=256):

        super().__init__()
        self.fc = nn.Linear(in_features, base_ch * 4 * 4)
        self.up1 = DeconvBlock(base_ch, base_ch // 2)  # 4 -> 8
        self.up2 = DeconvBlock(base_ch // 2, base_ch // 4)  # 8 -> 16
        self.up3 = DeconvBlock(base_ch // 4, base_ch // 8)  # 16 -> 32
        # Final RGB projection
        self.to_rgb = nn.Conv2d(base_ch // 8, 3, kernel_size=3, padding=1)

    def forward(self, x, *args, **kwargs):
        B, L = x.shape[0], x.shape[1]
        x = self.fc(x)  # (B, base_ch*4*4)
        x = x.view(B * L, -1, 4, 4)
        x = self.up1(x)  # (B, base_ch/2, 8, 8)
        x = self.up2(x)  # (B, base_ch/4, 16, 16)
        x = self.up3(x)  # (B, base_ch/8, 32, 32)
        x = torch.sigmoid(self.to_rgb(x))  # (B, 3, 32, 32)
        x = x.view(B, L, *x.shape[1:])
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, width=None, norm='batch'):
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

class ImageNetDeconvDecoderLarge(nn.Module):
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
        B, L = x.shape[0], x.shape[1]
        x = self.fc(x)  # (B, base_ch*4*4)
        x = x.view(B * L, -1, 4, 4)
        x = self.up1(x)  # (B, base_ch/2, 8, 8)
        x = self.up2(x)  # (B, base_ch/4, 16, 16)
        x = self.up3(x)  # (B, base_ch/8, 32, 32)
        x = self.up4(x)  # (B, base_ch/8, 32, 32)
        x = self.up5(x)  # (B, base_ch/8, 32, 32)
        x = self.up6(x)  # (B, base_ch/8, 32, 32)
        x = torch.sigmoid(self.to_rgb(x))  # (B, 3, 32, 32)
        x = x.view(B, L, *x.shape[1:])
        return x

class ImageNetDeconvDecoder(nn.Module):
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
        B, L = x.shape[0], x.shape[1]
        x = self.fc(x)  # (B, base_ch*4*4)
        x = x.view(B * L, -1, 4, 4)
        x = self.up1(x)  # (B, base_ch/2, 8, 8)
        x = self.up2(x)  # (B, base_ch/4, 16, 16)
        x = self.up3(x)  # (B, base_ch/8, 32, 32)
        x = self.up4(x)  # (B, base_ch/8, 32, 32)
        x = self.up5(x)  # (B, base_ch/8, 32, 32)

        x = torch.sigmoid(self.to_rgb(x))  # (B, 3, 32, 32)
        x = x.view(B, L, *x.shape[1:])
        return x

class SSMStateDeconvDecoder(nn.Module):
    def __init__(self, in_features=(784, 8), base_ch=512, norm='batch'):
        super().__init__()

        self.fc1 = nn.Linear(in_features[0] * in_features[1], in_features[0] * in_features[1])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features[0] * in_features[1], base_ch * 4 * 4)
        self.relu2 = nn.ReLU()


        self.up1 = DeconvBlock(base_ch, base_ch // 2, norm=norm, width=8)  # 4 -> 8
        self.up2 = DeconvBlock(base_ch // 2, base_ch // 4, norm=norm, width=16)  # 8 -> 16
        self.up3 = DeconvBlock(base_ch // 4, base_ch // 8, norm=norm, width=32)  # 16 -> 32
        self.up4 = DeconvBlock(base_ch // 8, base_ch // 16, norm=norm, width=64)  # 32 -> 64
        self.up5 = DeconvBlock(base_ch // 16, base_ch // 32, norm=norm, width=128)  # 64 -> 128

        # Final RGB projection
        self.to_rgb = nn.Conv2d(base_ch // 32, 3, kernel_size=3, padding=1)

    def forward(self, x, *args, **kwargs):
        x = torch.flatten(x, start_dim=1)
        x = self.relu1(self.fc1(x)) # (B, base_ch*4*4)
        x = self.relu2(self.fc2(x)) # (B, base_ch*4*4)
        x =x.reshape(x.shape[0], -1, 4, 4)


        x = self.up1(x)  # (B, base_ch/2, 8, 8)
        x = self.up2(x)  # (B, base_ch/4, 16, 16)
        x = self.up3(x)  # (B, base_ch/8, 32, 32)
        x = self.up4(x)  # (B, base_ch/8, 32, 32)
        x = self.up5(x)  # (B, base_ch/8, 32, 32)

        x = torch.sigmoid(self.to_rgb(x))  # (B, 3, 32, 32)
        return x


if __name__ == '__main__':
    # decoder = UpConvReconHead(392, 900, 300, (1, 300, 300))
    # pos_recon_head = PosReconHead(392, 786, 25, (5, 5))
    # dec = ImageNetDeconvDecoderLarge(in_features=392, base_ch=512, norm='rms')

    # dec = ImageNetDeconvDecoderLarge(in_features=392, base_ch=512, norm='rms')
    # a = torch.rand(size=(5, 8, 392))
    # b = dec(a)

    dec = SSMStateDeconvDecoder(in_features=(784, 8), base_ch=512, norm='rms')
    a = torch.rand(size=(5, 784, 8))
    b = dec(a)

    # pos_recon_head = UpConvReconCifarHeadLarge(392)
    #
    # a = torch.rand(size=(5, 8, 392))
    # x_pos = torch.randint(low=-10, high=10, size=(16, 8))
    # y_pos = torch.randint(low=-10, high=10, size=(16, 8))
    # g = pos_recon_head(a, x_pos, y_pos)

    # b = decoder(a)

