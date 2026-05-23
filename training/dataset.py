import numpy as np
import torch
from torch.utils.data import Dataset


class ARDataset(Dataset):
    def __init__(
        self,
        temp_formation,
        temp_frac,
        pres_formation,
        pres_frac,
        action,
        aux_energy_bhp,
        k_max=5,
        *,
        heterogeneous=False,
        por_matrix=None,
        por_frac=None,
        perm_matrix=None,
        perm_frac=None,
    ):
        self.temp_formation = temp_formation
        self.temp_frac = temp_frac
        self.pres_formation = pres_formation
        self.pres_frac = pres_frac
        self.action = action
        self.aux_energy_bhp = aux_energy_bhp
        self.heterogeneous = heterogeneous

        self.temp_min = 20.0
        self.temp_max = 185.0
        self.action_min = 0.0
        self.action_max = 5000.0
        self.energy_max = 2.9e12

        if heterogeneous:
            assert por_matrix is not None, "hetero mode requires por_matrix"
            self.por_matrix = por_matrix
            self.por_frac = por_frac
            self.perm_matrix = perm_matrix
            self.perm_frac = perm_frac
            self.pres_min = 1300.0
            self.pres_max = 70000.0
            self.por_min_matrix = 0.03
            self.por_max_matrix = 0.07
            self.por_min_frac = 0.002
            self.por_max_frac = 0.008
            self.perm_min_matrix = 0.05
            self.perm_max_matrix = 0.12
            self.perm_min_frac = 3.0
            self.perm_max_frac = 190.0
        else:
            self.pres_min = 1900.0
            self.pres_max = 68000.0

        self._temp_range = self.temp_max - self.temp_min
        self._pres_range = self.pres_max - self.pres_min
        self._action_range = self.action_max - self.action_min
        self._energy_log_denom = np.log1p(self.energy_max)

        self.k_max = k_max
        self.n_trajectories = len(temp_formation)
        self.n_timesteps = 156

    def __len__(self):
        return self.n_trajectories

    def _to_tensor(self, np_array):
        return torch.from_numpy(np_array.copy())

    def _state_at(self, idx, step):
        tf = (self._to_tensor(self.temp_formation[idx, step]) - self.temp_min) / self._temp_range
        tfr = (self._to_tensor(self.temp_frac[idx, step]) - self.temp_min) / self._temp_range
        pf = (self._to_tensor(self.pres_formation[idx, step]) - self.pres_min) / self._pres_range
        pfr = (self._to_tensor(self.pres_frac[idx, step]) - self.pres_min) / self._pres_range
        return torch.stack([tf, tfr, pf, pfr], dim=0).float()

    def _action_at(self, idx, step):
        a = (self._to_tensor(self.action[idx, step]) - self.action_min) / self._action_range
        return a.float()

    def _aux_at(self, idx, step):
        aux = self._to_tensor(self.aux_energy_bhp[idx, step]).float()
        energy_rate = torch.log1p(aux[9:]) / self._energy_log_denom
        bhp = (aux[0:9] - self.pres_min) / self._pres_range
        return torch.cat([energy_rate, bhp], dim=0)

    def _static_at(self, idx):
        pm = (self._to_tensor(self.por_matrix[idx]) - self.por_min_matrix) / (self.por_max_matrix - self.por_min_matrix)
        pf = (self._to_tensor(self.por_frac[idx]) - self.por_min_frac) / (self.por_max_frac - self.por_min_frac)
        km = (self._to_tensor(self.perm_matrix[idx]) - self.perm_min_matrix) / (self.perm_max_matrix - self.perm_min_matrix)
        kf = (self._to_tensor(self.perm_frac[idx]) - self.perm_min_frac) / (self.perm_max_frac - self.perm_min_frac)
        return torch.stack([pm, pf, km, kf], dim=0).float()

    def __getitem__(self, idx):
        t = np.random.randint(0, self.n_timesteps)

        y_t = self._state_at(idx, t)
        y_tp1 = self._state_at(idx, t + 1)
        action_t = self._action_at(idx, t)
        aux_t = self._aux_at(idx, t)
        aux_tp1 = self._aux_at(idx, t + 1)

        y_history = []
        action_history = []
        valid_k = []
        for i in range(self.k_max):
            step = t - (self.k_max - i)
            if step >= 0:
                y_history.append(self._state_at(idx, step))
                action_history.append(self._action_at(idx, step))
                valid_k.append(1.0)
            else:
                y_history.append(torch.zeros_like(y_t))
                action_history.append(torch.zeros_like(action_t))
                valid_k.append(0.0)

        y_history = torch.stack(y_history, dim=0)
        action_history = torch.stack(action_history, dim=0)
        valid_k = torch.tensor(valid_k, dtype=torch.float)

        base = (y_history, action_history, valid_k, y_t, y_tp1, action_t, aux_t, aux_tp1)

        if self.heterogeneous:
            return base + (self._static_at(idx),)
        return base
