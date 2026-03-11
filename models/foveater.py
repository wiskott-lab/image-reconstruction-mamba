import torch
from torch.functional import F
from tools.pos_utils import get_2d_sincos_pos_embed
import config
import torch.nn as nn


class FoveationGrid(nn.Module):
    # grid overlay on image reducing action space

    def __init__(self, grid_axis_size: int, img_axis_size: int):
        super().__init__()
        self.num_actions = grid_axis_size ** 2
        self.grid_axis_size = grid_axis_size
        self.grid_point_diameter = img_axis_size // grid_axis_size
        self.grid_margin = self.grid_point_diameter // 2

    def grid_x_y_to_pixel_x_y(self, grid_x, grid_y):
        pixel_x = torch.full_like(input=grid_x, fill_value=self.grid_margin) + grid_x * self.grid_point_diameter
        pixel_y = torch.full_like(input=grid_y, fill_value=self.grid_margin) + grid_y * self.grid_point_diameter
        return pixel_x, pixel_y

    def get_random_action(self, n_actions):
        action = torch.randint(0, self.num_actions, (n_actions,))
        return action

    def get_scan_path(self):
        action = torch.arange(start=0, step=1, end=self.num_actions)
        pixel_x, pixel_y = self.action_to_pixel_x_y(action)
        return pixel_x, pixel_y

    def action_to_grid_x_y(self, action):
        grid_y = action // self.grid_axis_size
        grid_x = action % self.grid_axis_size
        return grid_x, grid_y

    def action_to_pixel_x_y(self, action):
        grid_x, grid_y = self.action_to_grid_x_y(action)
        pixel_x, pixel_y = self.grid_x_y_to_pixel_x_y(grid_x, grid_y)
        return pixel_x, pixel_y

    def get_random_pixel_x_y(self, n_pixels):
        action = self.get_random_action(n_actions=n_pixels)
        pixel_x, pixel_y = self.action_to_pixel_x_y(action=action)
        return pixel_x, pixel_y


class Foveater(nn.Module):
    def __init__(self, emb_dim, patch_sizes, img_axis_size, pad_value=0.0, use_pos_emb=True,
                 img_channels=1):  # pos_emb = x, y
        super().__init__()
        self.patch_sizes = patch_sizes
        self.resize_to = patch_sizes[0]
        self.pad_value = pad_value
        self.img_axis_size = img_axis_size
        self.embedder = nn.Linear(in_features=patch_sizes[0] ** 2 * len(patch_sizes) * img_channels,
                                  out_features=emb_dim)
        self.pad_margin = self.patch_sizes[-1] // 2
        self.use_pos_emb = use_pos_emb
        self.pos_emb = get_2d_sincos_pos_embed(H=img_axis_size, W=img_axis_size, dim=emb_dim)

    def foveate(self, img, pixel_x, pixel_y):
        resized_patches = []
        img = F.pad(img, (self.pad_margin, self.pad_margin, self.pad_margin, self.pad_margin), mode="constant",
                    value=self.pad_value)
        for patch_size in self.patch_sizes:
            patch = crop(img=img, pixel_x=pixel_x + self.pad_margin, pixel_y=pixel_y + self.pad_margin,
                         crop_size=patch_size)
            resized_patches.append(
                F.interpolate(patch, size=self.resize_to, mode='bilinear', align_corners=True, antialias=True))
        foveation_token = torch.concat(resized_patches, dim=1).flatten(start_dim=1)
        mamba_token = self.embedder(foveation_token)
        if self.use_pos_emb:
            pos_token = self.pos_emb[pixel_y, pixel_x, :].to(config.DEVICE)
            mamba_token = mamba_token + pos_token
        return mamba_token


class Foveater2(nn.Module):
    def __init__(self, backbones, emb_dim, patch_sizes, img_axis_size, pad_value=0.0, use_pos_emb=True,
                 img_channels=1):  # pos_emb = x, y
        super().__init__()
        self.patch_sizes = patch_sizes
        self.resize_to = patch_sizes[0]
        self.pad_value = pad_value
        self.img_axis_size = img_axis_size
        self.pad_margin = self.patch_sizes[-1] // 2
        self.use_pos_emb = use_pos_emb
        self.pos_emb = get_2d_sincos_pos_embed(H=img_axis_size, W=img_axis_size, dim=emb_dim)
        self.backbones = backbones

    def foveate(self, img, pixel_x, pixel_y):
        patches = []
        img = F.pad(img, (self.pad_margin, self.pad_margin, self.pad_margin, self.pad_margin), mode="constant",
                    value=self.pad_value)
        for i in range(len(self.patch_sizes)):
            patch = crop(img=img, pixel_x=pixel_x + self.pad_margin, pixel_y=pixel_y + self.pad_margin,
                         crop_size=self.patch_sizes[i])
            patch = F.interpolate(patch, size=self.resize_to, mode='bilinear', align_corners=True, antialias=True)
            patches.append(patch)
        token = self.backbones(patches)

        if self.use_pos_emb:
            pos_token = self.pos_emb[pixel_y, pixel_x, :].to(config.DEVICE)
            token = token + pos_token
        return token


class Foveater3(nn.Module):
    def __init__(self, emb_dim, patch_sizes, img_axis_size, use_pos_emb=True, temperature=10000):  # pos_emb = x, y
        super().__init__()
        self.patch_sizes = patch_sizes
        self.resize_to = patch_sizes[0]
        self.img_axis_size = img_axis_size
        # self.embedder = nn.Linear(in_features=patch_sizes[0] ** 2 * len(patch_sizes) * img_channels, out_features=emb_dim)
        self.use_pos_emb = use_pos_emb
        self.current_x_pos, self.current_y_pos = None, None
        self.omega = torch.arange(emb_dim // 4, dtype=torch.float32, device=config.DEVICE)
        self.omega = 1. / (temperature ** (self.omega / (emb_dim / 4)))

    def foveate(self, img, pixel_x, pixel_y):
        if self.current_x_pos is None:
            self.current_x_pos = torch.zeros(size=(img.shape[0], 2))
            self.current_y_pos = torch.zeros(size=(img.shape[0], 2))

        self.current_x_pos += pixel_x
        self.current_y_pos += pixel_y

        resized_patches = []
        # img = F.pad(img, (self.pad_margin, self.pad_margin, self.pad_margin, self.pad_margin), mode="constant", value=self.pad_value)

        foveation_token = torch.concat(resized_patches, dim=1).flatten(start_dim=1)

        mamba_token = self.embedder(foveation_token)
        if self.use_pos_emb:
            pos_token = get_pos_embeddings(x_pos=self.current_x_pos, y_pos=self.current_y_pos, omega=self.omega).to(
                config.DEVICE)
            mamba_token = mamba_token + pos_token
        return mamba_token


class FoveationProperties:

    def __init__(self, patch_sizes, resize_sizes):
        self.patch_sizes = patch_sizes
        self.resize_sizes = resize_sizes



def get_pos_embeddings(x_pos, y_pos, omega):
    out_x = x_pos[..., None] * omega[None, :]
    sin_x, cos_x = torch.sin(out_x), torch.cos(out_x)
    out_y = y_pos[..., None] * omega[None, :]
    sin_y, cos_y = torch.sin(out_y), torch.cos(out_y)
    pos_emb = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)
    return pos_emb


def crop(img, pixel_x, pixel_y, crop_size):
    top, left = pixel_y - crop_size // 2, pixel_x - crop_size // 2
    bottom, right = top + crop_size, left + crop_size
    cropped_tensor = torch.stack([img[i, :, top[i]:bottom[i], left[i]:right[i]] for i in range(img.shape[0])])
    # TODO vectorize
    return cropped_tensor


if __name__ == '__main__':
    img = torch.rand(size=(10, 1, 48, 57))

    pass
    foveater = Foveater(emb_dim=4, patch_sizes=(3,), pad_value=0, img_axis_size=28)
    # batch = torch.ones(size=(10, 1, 28, 28))
    # l = foveater.get_scan_path_seq(batch)
    # g = foveater.get_rnd_seq(batch, 400)
    #
    # pos_x = torch.randint(0, 28, (batch.shape[0],))
    # pos_y = torch.randint(0, 28, (batch.shape[0],))
    # u = foveater.foveate(batch, pos_x=pos_x, pos_y=pos_y)
    # pass
