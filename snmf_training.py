import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import os


class TrainConfig:
    def __init__(self):
        # Basic config
        self.data_folder = "precomputed_activations"
        self.checkpoints_folder = "trained_saes"
        self.plots_folder = "plots"
        self.cache_dir = "/home/can/models"

        self.data_path = "precomputed_activations_debug/lm_activations_0_of_1_tokens2000.pt"

        self.device = "cuda"
        self.dtype = torch.bfloat16

        self.batch_size = 20
        self.num_total_tokens = 2000
        self.num_total_batches = self.num_total_tokens // self.batch_size
        self.num_test_tokens = int(self.num_total_tokens * 0.01)
        self.shuffle_train = True
        
        self.llm_hidden_dim = 2304
        self.mlp_hidden_dim = self.llm_hidden_dim * 4
        self.expansion_factor = 2
        self.sae_hidden_dim = self.expansion_factor * self.mlp_hidden_dim

    def to_dict(self):
        save_dict = self.__dict__.copy()
        for key in ["dtype"]:
            save_dict[key] = str(save_dict[key])

        return save_dict

    @classmethod
    def from_dict(cls, save_path):
        cfg = cls()
        save_dict = json.load(open(save_path))
        cfg.__dict__.update(save_dict)

        if "dtype" in save_dict:
            if save_dict["dtype"] == "torch.bfloat16":
                cfg.dtype = torch.bfloat16
            elif save_dict["dtype"] == "torch.float32":
                cfg.dtype = torch.float32
            else:
                raise ValueError(f"Invalid dtype: {save_dict['dtype']}")

        return cfg


def ensure_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# D: hidden dim of MLP
# S: hidden dim of snmf
# b: total number of tokens

class SNMF(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg

        self.Z_DS = torch.randn(cfg.mlp_hidden_dim, cfg.sae_hidden_dim, device=cfg.device, dtype=cfg.dtype) # Only closed form updates
        self.Y_Sb = torch.rand(cfg.sae_hidden_dim, cfg.num_total_tokens, device=cfg.device, dtype=cfg.dtype) # non-negative SGD updates


def load_precomputed_activations(cfg: TrainConfig):
    activations = torch.load(cfg.data_path)

    activations = activations.to(cfg.device)
    activations = activations.to(cfg.dtype)

    print(f"Loaded activations of shape {activations.shape}")
    return activations


def zero_negative_entries(tensor: torch.Tensor, flip_sign: bool = False):
    mask = tensor < 0
    if flip_sign:
        mask = ~mask

    out = torch.where(mask, torch.zeros_like(tensor), tensor)
    return out

def reconstruction_error(snmf: SNMF, A_bD: torch.Tensor):
    return torch.norm(A_bD.T - snmf.Z_DS @ snmf.Y_Sb, "fro")

def train_snmf(cfg: TrainConfig, snmf: SNMF):
    act_BPD = load_precomputed_activations(cfg)
    A_bD = act_BPD.reshape(-1, act_BPD.shape[-1])

    num_steps = 10

    for step in range(num_steps):
        # Update Z closed form exact
        Y_Sb_inv = torch.linalg.inv((snmf.Y_Sb @ snmf.Y_Sb.T).float()).to(cfg.dtype)
        Z_new = A_bD.T @ snmf.Y_Sb.T @ Y_Sb_inv
        snmf.Z_DS = Z_new

        # Update Y SGD multiplicative non-negative update minimizing frobenius norm
        ZT_A_plus_Sb = F.relu(snmf.Z_DS.T @ A_bD.T)
        ZT_A_minus_Sb = F.relu(-snmf.Z_DS.T @ A_bD.T)
        ZT_Z_plus_SS = F.relu(snmf.Z_DS.T @ snmf.Z_DS)
        ZT_Z_minus_SS = F.relu(-snmf.Z_DS.T @ snmf.Z_DS)


        encouraged_updates = ZT_A_plus_Sb + ZT_Z_minus_SS @ snmf.Y_Sb
        discouraged_updates = ZT_A_minus_Sb + ZT_Z_plus_SS @ snmf.Y_Sb
        
        Y_new = snmf.Y_Sb * torch.sqrt(encouraged_updates / (discouraged_updates + 1e-8))
        snmf.Y_Sb = Y_new

        print(f"Y_Sb isnan: {torch.isnan(snmf.Y_Sb).any()}")

        loss = reconstruction_error(snmf, A_bD)
        print(f"Reconstruction error: {loss}")



if __name__ == "__main__":
    cfg = TrainConfig()
    snmf = SNMF(cfg)
    train_snmf(cfg, snmf)
