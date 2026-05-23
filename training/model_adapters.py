import torch


class SingleTensorAdapter:
    """For UNet3D and FNOWrapper: concatenate all inputs into one tensor."""

    def __init__(self, heterogeneous=False):
        self.heterogeneous = heterogeneous

    def build_model_input(self, y, action, static=None):
        rate = action.unsqueeze(1)
        mask = (rate != 0).float()
        if self.heterogeneous and static is not None:
            return torch.cat([y, static, rate, mask], dim=1)
        return torch.cat([y, rate, mask], dim=1)

    def forward(self, model, model_input):
        return model(model_input)


class DualTensorAdapter:
    """For LOGLO_FNO: separate state and action tensors."""

    def __init__(self, heterogeneous=False):
        self.heterogeneous = heterogeneous

    def build_model_input(self, y, action, static=None):
        rate = action.unsqueeze(1)
        mask = (rate != 0).float()
        if self.heterogeneous and static is not None:
            state_input = torch.cat([y, static], dim=1)
            action_input = torch.cat([rate, mask, static], dim=1)
        else:
            state_input = y
            action_input = torch.cat([rate, mask], dim=1)
        return (state_input, action_input)

    def forward(self, model, model_input):
        state_input, action_input = model_input
        return model(state_input, action_input)


def create_adapter(model_type, heterogeneous):
    if model_type in ('unet3d', 'fno'):
        return SingleTensorAdapter(heterogeneous=heterogeneous)
    elif model_type == 'loglo':
        return DualTensorAdapter(heterogeneous=heterogeneous)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
