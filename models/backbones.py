import torch
import torch.nn as nn



class SimpleJoiner(nn.Module):
    def __init__(self, in_features, out_features=128):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, patches: list):
        token = torch.concat(patches, dim=-1)
        output = self.linear_1(token)
        return output


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


class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten_1 = nn.Flatten(start_dim=-3)

    def forward(self, x):
        x = self.flatten_1(x)
        return x


class Backbone(nn.Module):
    def __init__(self, dropout_p=0.0, out_features=128):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
        self.dropout_1 = nn.Dropout2d(dropout_p)
        self.bn_1 = nn.BatchNorm2d(32)
        self.relu_1 = nn.ReLU()  # 64x32x32

        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.dropout_2 = nn.Dropout2d(dropout_p)
        self.bn_2 = nn.BatchNorm2d(64)
        self.relu_2 = nn.ReLU()  # 64x32x32

        self.conv_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.dropout_3 = nn.Dropout2d(dropout_p)
        self.bn_3 = nn.BatchNorm2d(128)
        self.relu_3 = nn.ReLU()  # 64x32x32

        self.flatten_4 = nn.Flatten(start_dim=1)
        self.linear_4 = nn.Linear(in_features=128, out_features=out_features)
        self.relu_4 = nn.ReLU()


    def forward(self, x):
        x = self.relu_1(self.bn_1(self.dropout_1(self.conv_1(x))))
        x = self.relu_2(self.bn_2(self.dropout_2(self.conv_2(x))))
        x = self.relu_3(self.bn_3(self.dropout_3(self.conv_3(x))))
        x = self.relu_4(self.linear_4(self.flatten_4(x)))
        return x




if __name__ == '__main__':
    simple_bb = SimpleBackbone()
    u1 = torch.rand(size=(10, 3, 7, 7))
    u2 = simple_bb(u1)

    # bb = Joiner()
    # a = [torch.rand(size=(10, 3, 7, 7)) for _ in range(1)]
    # b = bb(a)

