import torch

import config


class ActionConverter:
    # grid overlay on image reducing action space

    def __init__(self, distances=(1, 2, 5, 10, 20, 50)):
        distances = torch.Tensor(list(distances)).int()
        self.two_way_distances = torch.concat([-distances.flip(0), torch.tensor([0]), distances]).to(config.DEVICE)
        self.axis_size = self.two_way_distances.shape[0]
        self.num_actions = self.two_way_distances.shape[0] ** 2

    def convert(self, action):
        grid_y = action // self.axis_size
        grid_x = action % self.axis_size
        shift_x = self.two_way_distances[grid_x]
        shift_y = self.two_way_distances[grid_y]
        return shift_x, shift_y

class EightAngularActionConverter:
    # grid overlay on image reducing action space

    def __init__(self, distances=(1, 5, 20, 75)):
        self.distances = torch.Tensor(list(distances)).int().to(config.DEVICE)

        # self.two_way_distances = torch.concat([-distances.flip(0), torch.tensor([0]), distances]).to(config.DEVICE)
        # self.axis_size = self.two_way_distances.shape[0]
        self.num_actions = len(distances) * 8
        self.x_factor = torch.tensor([1, 1, 0, -1, -1, -1, 0, 1], device=config.DEVICE)
        self.y_factor = torch.tensor([0, 1, 1, 1, 0, -1, -1, -1], device=config.DEVICE)


    def convert(self, action):
        circle = action // 8
        angle = action % 8
        shift_x = self.x_factor[angle] * self.distances[circle]
        shift_y =  self.y_factor[angle] * self.distances[circle]
        return shift_x, shift_y


if __name__ == '__main__':
    action_to_pos = EightAngularActionConverter()
    action_to_pos.convert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    pass



