import random

import torch
from torch.functional import F
import config
from models.foveater import FoveationProperties
import math




def get_query_pos(fov_props, img, seq_len=None, seq_type='rnd', sub_img_size=None):
    if sub_img_size is None:
        x_range = (fov_props.patch_sizes[-1] // 2, img.shape[-1] - fov_props.patch_sizes[-1] // 2)
        y_range = (fov_props.patch_sizes[-1] // 2, img.shape[-2] - fov_props.patch_sizes[-1] // 2)
    else:
        x_range = (sub_img_size // 2, img.shape[-1] - sub_img_size // 2)
        y_range = (sub_img_size // 2, img.shape[-2] - sub_img_size // 2)
    if seq_type == 'rnd':
        pos_x = torch.randint(low=x_range[0], high=x_range[1], size=(img.shape[0], seq_len)).to(config.DEVICE)
        pos_y = torch.randint(low=y_range[0], high=y_range[1], size=(img.shape[0], seq_len)).to(config.DEVICE)
    else:
        raise NameError(f'Unknown sequence type: {seq_type}')
    rel_pos_x, rel_pos_y = abs_pos_to_rel_pos(img, pos_x, pos_y)
    return rel_pos_x, rel_pos_y


def get_seq(fov_props, img, seq_len=None, seq_type='rnd', pos_x=None, pos_y=None, x_range=None, y_range=None):
    sequence = []
    if pos_x is None or pos_y is None:
        if x_range is None:
            x_range = (fov_props.patch_sizes[-1] // 2, img.shape[-1] - fov_props.patch_sizes[-1] // 2)
        if y_range is None:
            y_range = (fov_props.patch_sizes[-1] // 2, img.shape[-2] - fov_props.patch_sizes[-1] // 2)
        if seq_type == 'rnd':
            pos_x = torch.randint(low=x_range[0], high=x_range[1], size=(img.shape[0], seq_len)).to(config.DEVICE)
            pos_y = torch.randint(low=y_range[0], high=y_range[1], size=(img.shape[0], seq_len)).to(config.DEVICE)
        elif seq_type == 'rnd_normal':
            pos_x = torch.normal( mean=63.5,std=21.3, size=(img.shape[0], seq_len),device=config.DEVICE).round().long().clamp(x_range[0], x_range[1] - 1)
            pos_y = torch.normal(mean=63.5, std=21.3, size=(img.shape[0], seq_len), device=config.DEVICE).round().long().clamp(y_range[0], y_range[1] - 1)
        elif seq_type == 'rnd_ordered':
            pos_x_1 = torch.randint(low=x_range[0], high=x_range[1] // 2, size=(img.shape[0], math.ceil(seq_len / 4))).to(config.DEVICE)
            pos_x_2 = torch.randint(low=x_range[1] // 2, high=x_range[1], size=(img.shape[0], math.ceil(seq_len / 4))).to(config.DEVICE)
            pos_x_3 = torch.randint(low=x_range[0], high=x_range[1] // 2, size=(img.shape[0], math.ceil(seq_len / 4))).to(config.DEVICE)
            pos_x_4 = torch.randint(low=x_range[1] // 2, high=x_range[1], size=(img.shape[0], math.ceil(seq_len / 4))).to(config.DEVICE)

            pos_y_1 = torch.randint(low=y_range[0], high=y_range[1] // 2, size=(img.shape[0], math.ceil(seq_len / 4))).to(config.DEVICE)
            pos_y_2 = torch.randint(low=y_range[1] // 2, high=y_range[1], size=(img.shape[0], math.ceil(seq_len / 4))).to(config.DEVICE)
            pos_y_3 = torch.randint(low=y_range[0], high=y_range[1] // 2, size=(img.shape[0], math.ceil(seq_len / 4))).to(config.DEVICE)
            pos_y_4 = torch.randint(low=y_range[1] // 2, high=y_range[1], size=(img.shape[0], math.ceil(seq_len / 4))).to(config.DEVICE)
            pos_x, pos_y = [pos_x_1, pos_x_2, pos_x_3, pos_x_4], [pos_y_1, pos_y_2, pos_y_3, pos_y_4]
            random.shuffle(pos_x), random.shuffle(pos_y)
            pos_x, pos_y = torch.cat(pos_x, dim=1), torch.cat(pos_y, dim=1)

        elif seq_type == 'full':
            pos_x = torch.arange(start=x_range[0], end=x_range[1]).to(config.DEVICE)
            pos_y = torch.arange(start=y_range[0], end=y_range[1]).to(config.DEVICE)
            cartesian = torch.cartesian_prod(pos_x, pos_y)
            pos_x, pos_y = cartesian[:, 0].expand(img.shape[0], -1), cartesian[:, 1].expand(img.shape[0], -1)
        elif seq_type == 'relevant_only':
            pos_x = torch.arange(start=x_range[0], end=x_range[1] - 1, step=fov_props.patch_sizes[0]).to(config.DEVICE)
            pos_y = torch.arange(start=y_range[0], end=y_range[1] - 1, step=fov_props.patch_sizes[0]).to(config.DEVICE)
            pos_x = torch.cat([pos_x, torch.tensor([x_range[1]]).to(config.DEVICE) - 1])
            pos_y = torch.cat([pos_y, torch.tensor([y_range[1]]).to(config.DEVICE) - 1])
            cartesian = torch.cartesian_prod(pos_x, pos_y)
            pos_x, pos_y = cartesian[:, 0].expand(img.shape[0], -1), cartesian[:, 1].expand(img.shape[0], -1)
        elif seq_type == 'in_a_row':
            pos_x = torch.arange(start=x_range[0], end=x_range[1]).to(config.DEVICE)
            pos_y = torch.arange(start=y_range[0], end=y_range[1]).to(config.DEVICE)
            cartesian = torch.cartesian_prod(pos_x, pos_y)
            pos_x, pos_y = cartesian[:, 0].expand(img.shape[0], -1), cartesian[:, 1].expand(img.shape[0], -1)
            pos_y = pos_y[:, :seq_len]
            pos_x = pos_x[:, :seq_len]
        else:
            raise NameError(f'Unknown sequence type: {seq_type}')
    for patch_size, resize_to in zip(fov_props.patch_sizes, fov_props.resize_sizes):
        patches = crop(img=img, pixel_x=pos_x, pixel_y=pos_y, crop_size=fov_props.patch_sizes[0])
        patches_view = patches.view(patches.shape[0] * patches.shape[1], -1, patches.shape[3], patches.shape[4])
        patches_view = F.interpolate(patches_view, size=resize_to, mode='bilinear', align_corners=True, antialias=True)
        sequence.append(
            patches_view.view(patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3], patches.shape[4]))
    rel_pos_x, rel_pos_y = abs_pos_to_rel_pos(img, pos_x, pos_y)
    return sequence, rel_pos_x, rel_pos_y


def abs_pos_to_rel_pos(img, abs_x, abs_y):
    origin_x, origin_y = img.shape[-1] // 2, img.shape[-2] // 2
    rel_x, rel_y = abs_x - origin_x, abs_y - origin_y
    return rel_x, rel_y



def crop(img, pixel_x, pixel_y, crop_size):
    B, C, H, W = tuple(img.shape)
    L = pixel_y.shape[1]
    top, left = pixel_y - crop_size // 2, pixel_x - crop_size // 2
    offsets = torch.arange(crop_size).to(config.DEVICE)
    rows = top[:, :, None] + offsets[None, :]  # (B, L, crop_size)
    cols = left[:, :, None] + offsets[None, :]  # (B, L, crop_size)
    rows = rows[:, :, None, :, None].expand(B, L, C, crop_size, crop_size)
    cols = cols[:, :, None, None, :].expand(B, L, C, crop_size, crop_size)
    batch_idx = torch.arange(B)[:, None, None, None, None].expand(B, L, C, crop_size, crop_size)
    channel_idx = torch.arange(C)[None, None, :, None, None].expand(B, L, C, crop_size, crop_size)
    seq_idx = torch.arange(L)[None, :, None, None, None].expand(B, L, C, crop_size, crop_size)
    img = img[:, None, :, :, :].expand(B, L, C, H, W)
    cropped_tensor = img[batch_idx, seq_idx, channel_idx, rows, cols]
    return cropped_tensor


if __name__ == '__main__':
    foveation_props = FoveationProperties(patch_sizes=(4,), resize_sizes=(4,))
    img = torch.rand(size=(10, 1, 128, 128))
    seq_len = 5
    seq_1, pos_x_1, pos_y_1 = get_seq(fov_props=foveation_props, seq_type='rnd', img=img, seq_len=8)
    print(torch.max(pos_x_1), torch.max(pos_y_1))



