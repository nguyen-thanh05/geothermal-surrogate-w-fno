import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device():
    if torch.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_action_for_mbe(action):
    rate = action.unsqueeze(1)
    mask = (rate != 0).float()
    return torch.cat([rate, mask], dim=1)


@torch.no_grad()
def update_ema(ema_model, model, ema_aux, aux_model, decay):
    for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
        ema_p.lerp_(model_p, 1.0 - decay)
    for ema_b, model_b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(model_b)
    for ema_p, model_p in zip(ema_aux.parameters(), aux_model.parameters()):
        ema_p.lerp_(model_p, 1.0 - decay)
    for ema_b, model_b in zip(ema_aux.buffers(), aux_model.buffers()):
        ema_b.copy_(model_b)


def capture_rng_states(loader_gen):
    state = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_cpu': torch.random.get_rng_state(),
        'loader_gen': loader_gen.get_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_states(rng_states, loader_gen):
    random.setstate(rng_states['python_random'])
    np.random.set_state(rng_states['numpy_random'])
    torch.random.set_rng_state(rng_states['torch_cpu'])
    loader_gen.set_state(rng_states['loader_gen'])
    if 'torch_cuda' in rng_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_states['torch_cuda'])
