import torch

import config


def get_2d_sincos_pos_embed(H, W, dim, temperature=10000):
    """
    H: height of grid
    W: width of grid
    dim: total embedding dimension (must be divisible by 4)
    Returns: [H * W, dim] positional embedding
    """
    assert dim % 4 == 0, "dim must be divisible by 4"
    dim_h = dim_w = dim // 2

    # generate position indices
    grid_y = torch.arange(H, dtype=torch.float32, device=config.DEVICE)
    grid_x = torch.arange(W, dtype=torch.float32, device=config.DEVICE)
    grid = torch.meshgrid(grid_y, grid_x, indexing="ij")  # [2, H, W]

    pos_y = get_1d_sin_cos_pos_embed(grid[0], dim_h, temperature)
    pos_x = get_1d_sin_cos_pos_embed(grid[1], dim_w, temperature)

    pos = torch.cat([pos_y, pos_x], dim=-1)
    # return pos.view(H * W, dim)
    return pos

def get_1d_sin_cos_pos_embed(pos, dim, temperature=10000):
    """
    pos: [H, W] tensor of positions
    dim: dimension for this axis
    """
    omega = torch.arange(dim // 2, dtype=torch.float32, device=config.DEVICE)
    omega = 1. / (temperature ** (omega / (dim / 2)))
    out = pos[..., None] * omega[None, :]
    sin = torch.sin(out)
    cos = torch.cos(out)
    return torch.cat([sin, cos], dim=-1)


if __name__ == '__main__':
    # get_2d_sincos_pos_embed(H=28, W=28, dim=128)
    a = torch.Tensor([[1, -1, 2, 3]])
    b = torch.Tensor([[1, -1, 2, 3]])
    a1 = get_1d_sin_cos_pos_embed(a, dim=64)
    b1 = get_1d_sin_cos_pos_embed(b, dim=64)


    b = torch.Tensor([2, 2])
    b1 = get_1d_sin_cos_pos_embed(a, dim=4)
    c = torch.Tensor([-1, -1])
    c1 = get_1d_sin_cos_pos_embed(a, dim=4)
    d = torch.Tensor([-2, -2])
    d1 = get_1d_sin_cos_pos_embed(a, dim=4)

