"""
StoxLSTM model implementation.
Stochastic latent recurrent model for time series forecasting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg

def reparam(mu, std):
    """Reparameterization trick for sampling from normal distribution."""
    eps = torch.randn_like(std)
    return mu + std * eps

def kl_normal(qm, qs, pm, ps):
    """KL divergence between two normal distributions."""
    # KL(N(qm,qs^2) || N(pm,ps^2))
    return torch.log(ps / qs + 1e-12) + (qs ** 2 + (qm - pm) ** 2) / (2 * ps ** 2) - 0.5

class StoxCell(nn.Module):
    """Stochastic cell with GRU and latent variable."""
    
    def __init__(self, d_model: int, d_latent: int):
        super().__init__()
        self.gru = nn.GRUCell(input_size=d_model, hidden_size=d_model)
        # Prior over z_t
        self.prior_m = nn.Linear(d_model, d_latent)
        self.prior_s = nn.Linear(d_model, d_latent)
        # Emission head from [h_t, z_t]
        self.emit = nn.Sequential(
            nn.Linear(d_model + d_latent, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x_t, h_prev):
        # x_t: [B, d_model]
        h_t = self.gru(x_t, h_prev)  # [B, d_model]
        prior_m = self.prior_m(h_t)
        prior_s = F.softplus(self.prior_s(h_t)) + 1e-5
        return h_t, (prior_m, prior_s)

class PosteriorNet(nn.Module):
    """Bidirectional encoder that outputs q(z_t | ... ) params at patch granularity."""
    
    def __init__(self, d_model: int, d_latent: int):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=d_model, hidden_size=d_model // 2,
                              num_layers=1, bidirectional=True, batch_first=True)
        self.q_m = nn.Linear(d_model, d_latent)
        self.q_s = nn.Linear(d_model, d_latent)

    def forward(self, x_enc):  # [B, Np, d_model]
        h, _ = self.bilstm(x_enc)
        qm = self.q_m(h)  # [B, Np, d_latent]
        qs = F.softplus(self.q_s(h)) + 1e-5
        return qm, qs

class PatchEmbed(nn.Module):
    """Linear patch embedding (per channel)."""
    
    def __init__(self, P: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(P, d_model)

    def forward(self, patches):  # [B, Np, P, 1]
        x = patches.squeeze(-1)  # [B, Np, P]
        x = self.proj(x)  # [B, Np, d_model]
        return x

class StoxLSTMChannel(nn.Module):
    """
    Channel-independent pipeline for a single channel.
    Operates on patches of normalized values.
    """
    
    def __init__(self, P: int, d_model: int, d_latent: int, dropout=0.0):
        super().__init__()
        self.embed = PatchEmbed(P, d_model)
        self.cell = StoxCell(d_model, d_latent)
        self.posterior = PosteriorNet(d_model, d_latent)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)  # predict patch step mean (compact)

    def forward(self, patches_hist, patches_all, compute_posterior=True):
        B = patches_hist.size(0)
        x_hist = self.embed(patches_hist)  # [B, Np_hist, d_model]
        x_all = self.embed(patches_all) if compute_posterior else None

        if compute_posterior:
            qm_all, qs_all = self.posterior(self.drop(x_all))  # [B, Np_all, d_latent]
        else:
            qm_all = qs_all = None

        h = torch.zeros(B, x_hist.size(-1), device=x_hist.device)
        priors_m, priors_s, z_hist, h_seq, y_hist = [], [], [], [], []
        
        for t in range(x_hist.size(1)):
            x_t = x_hist[:, t, :]
            h, (pm, ps) = self.cell(x_t, h)
            priors_m.append(pm)
            priors_s.append(ps)
            
            if compute_posterior:
                z_t = reparam(qm_all[:, t, :], qs_all[:, t, :])
            else:
                z_t = reparam(pm, ps)
            z_hist.append(z_t)
            
            y_t = self.head(self.drop(self.cell.emit(torch.cat([h, z_t], dim=-1))))
            y_hist.append(y_t)
            h_seq.append(h)

        prior_m = torch.stack(priors_m, dim=1)  # [B, Np_hist, d_latent]
        prior_s = torch.stack(priors_s, dim=1)
        z_hist = torch.stack(z_hist, dim=1)  # [B, Np_hist, d_latent]
        h_seq = torch.stack(h_seq, dim=1)  # [B, Np_hist, d_model]
        y_hist = torch.stack(y_hist, dim=1).squeeze(-1)  # [B, Np_hist]

        return y_hist, (prior_m, prior_s), (qm_all, qs_all), h_seq

    @torch.no_grad()
    def forecast(self, patches_hist, Np_future: int):
        """Forecast future values using autoregressive prediction."""
        B = patches_hist.size(0)
        x_hist = self.embed(patches_hist)  # [B, Np_hist, d_model]

        h = torch.zeros(B, x_hist.size(-1), device=x_hist.device)
        outs = []

        # Roll through history (to warm up hidden)
        for t in range(x_hist.size(1)):
            x_t = x_hist[:, t, :]
            h, (pm, ps) = self.cell(x_t, h)
            z_t = reparam(pm, ps)
            y_t = self.head(self.cell.emit(torch.cat([h, z_t], dim=-1)))
            outs.append(y_t.squeeze(-1))

        # Forecast future via autoregressive prediction
        # Start with the last historical input
        x_prev = x_hist[:, -1, :]
        
        for t in range(Np_future):
            h, (pm, ps) = self.cell(x_prev, h)
            z_t = reparam(pm, ps)
            emission = self.cell.emit(torch.cat([h, z_t], dim=-1))
            y_t = self.head(emission)
            outs.append(y_t.squeeze(-1))
            
            # Use the emission output as input for next step (autoregressive)
            # This creates a feedback loop where predictions influence future predictions
            x_prev = emission

        return torch.stack(outs, dim=1)  # [B, Np_hist+Np_future]

class StoxLSTM_Multi(nn.Module):
    """Wrap multiple per-channel modules, sharing hyperparams but not weights."""
    
    def __init__(self, C: int, P: int, d_model: int, d_latent: int, dropout=0.0):
        super().__init__()
        self.C = C
        self.mods = nn.ModuleList([StoxLSTMChannel(P, d_model, d_latent, dropout) for _ in range(C)])

    def forward(self, patches_hist_list, patches_all_list):
        ys, priors, posts, hs = [], [], [], []
        for c in range(self.C):
            y, pr, po, h = self.mods[c](patches_hist_list[c], patches_all_list[c], compute_posterior=True)
            ys.append(y)
            priors.append(pr)
            posts.append(po)
            hs.append(h)
        return ys, priors, posts, hs

    @torch.no_grad()
    def forecast(self, patches_hist_list, Np_future: int):
        """Forecast future values for all channels."""
        outs = []
        for c in range(self.C):
            y = self.mods[c].forecast(patches_hist_list[c], Np_future)
            outs.append(y)
        return outs

def compute_elbo(ys, priors, posts, target_patch_vals, beta_kl=1.0):
    """Compute ELBO loss for training."""
    total_recon, total_kl = 0.0, 0.0
    
    for c in range(len(ys)):
        y = ys[c]  # [B, Nh]
        tgt = target_patch_vals[c]  # [B, Nh]
        recon = F.mse_loss(y, tgt, reduction='mean')
        
        pm, ps = priors[c]
        qm_all, qs_all = posts[c]
        Nh = pm.size(1)
        qm = qm_all[:, :Nh, :]
        qs = qs_all[:, :Nh, :]
        kl = kl_normal(qm, qs, pm, ps).sum(-1).mean()  # average over batch & time, sum latent dims
        
        total_recon = total_recon + recon
        total_kl = total_kl + kl
    
    elbo = total_recon + beta_kl * total_kl
    return elbo, total_recon.item(), total_kl.item()

def create_model(C=None, P=None, d_model=None, d_latent=None, dropout=None):
    """Create StoxLSTM model with given or default parameters."""
    if C is None:
        C = len(cfg.cols)
    if P is None:
        P = cfg.P
    if d_model is None:
        d_model = cfg.d_model
    if d_latent is None:
        d_latent = cfg.d_latent
    if dropout is None:
        dropout = cfg.dropout
    
    model = StoxLSTM_Multi(C=C, P=P, d_model=d_model, d_latent=d_latent, dropout=dropout)
    model = model.to(cfg.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created:")
    print(f"  Channels: {C}")
    print(f"  Patch size: {P}")
    print(f"  Model dim: {d_model}")
    print(f"  Latent dim: {d_latent}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {cfg.device}")
    
    return model
