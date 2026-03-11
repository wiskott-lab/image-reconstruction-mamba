import torch


def convert_state_dict(obj):
    return {key: value.cpu() if torch.is_tensor(value) else value for key, value in obj.state_dict().items()}

def generate_model_state(model):
    model = getattr(model, "_orig_mod", model)
    model_state = convert_state_dict(model)
    return model_state


if __name__ == '__main__':
    pass
