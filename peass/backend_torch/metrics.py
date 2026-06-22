"""
PEASS PyTorch Metrics and Auditory Similarity
File path: peass/backend_torch/metrics.py
"""
import torch
from .auditory_model import generate_auditory_internal_representation_torch


def calculate_bss_eval_energy_ratios(true_source, target_dist, interf, artif):
    eps = 1e-15
    # Flatten across time and channels
    s = true_source.view(-1)
    e_t = target_dist.view(-1)
    e_i = interf.view(-1)
    e_a = artif.view(-1)

    energy_true = torch.dot(s, s)
    energy_dist = torch.dot(e_t, e_t)
    energy_interf = torch.dot(e_i, e_i)
    energy_artif = torch.dot(e_a, e_a)

    sum_td = s + e_t
    sum_int = sum_td + e_i
    sum_all = e_t + e_i + e_a

    ISR = 10.0 * torch.log10(torch.clamp(energy_true, min=eps) / torch.clamp(energy_dist, min=eps))
    SIR = 10.0 * torch.log10(torch.clamp(torch.dot(sum_td, sum_td), min=eps) / torch.clamp(energy_interf, min=eps))
    SAR = 10.0 * torch.log10(torch.clamp(torch.dot(sum_int, sum_int), min=eps) / torch.clamp(energy_artif, min=eps))
    SDR = 10.0 * torch.log10(torch.clamp(energy_true, min=eps) / torch.clamp(torch.dot(sum_all, sum_all), min=eps))

    return ISR.item(), SIR.item(), SAR.item(), SDR.item()


def calculate_auditory_similarity_metric_torch(ref_rep, test_rep, fs) -> float:
    """Vectorized calculation of PEMO-Q assimilation similarity."""
    num_bands, num_samples, num_mods = ref_rep.shape

    # Threshold masking
    test_rep = test_rep.clone()
    mask = torch.abs(test_rep) < torch.abs(ref_rep)
    test_rep[mask] = 0.25 * ref_rep[mask] + 0.75 * test_rep[mask]

    frame_len = int(min(num_samples, 0.1 * fs))
    num_frames = num_samples // frame_len
    num_samples = num_frames * frame_len

    ref_frames = ref_rep[:, :num_samples, :].reshape(num_bands, num_frames, frame_len, num_mods)
    ref_frames = ref_frames.permute(1, 0, 2, 3).reshape(num_frames, -1, num_mods)

    test_frames = test_rep[:, :num_samples, :].reshape(num_bands, num_frames, frame_len, num_mods)
    test_frames = test_frames.permute(1, 0, 2, 3).reshape(num_frames, -1, num_mods)

    eps = 1e-15
    ref_mean = torch.mean(ref_frames, dim=1, keepdim=True)
    lref = ref_frames - ref_mean

    test_mean = torch.mean(test_frames, dim=1, keepdim=True)
    ltest = test_frames - test_mean

    denom = torch.sqrt(torch.sum(lref**2, dim=1) * torch.sum(ltest**2, dim=1))
    lpsm = torch.sum(lref * ltest, dim=1) / torch.clamp(denom, min=eps)

    lnms = torch.sum(test_frames**2, dim=1)
    local_p = torch.sum(lpsm * lnms, dim=1) / torch.clamp(torch.sum(lnms, dim=1), min=eps)

    # Global integration via cumulative sum
    test_sq = torch.sum(test_rep[:, :num_samples, :]**2, dim=(0, 2))
    cum_sum = torch.cat([torch.tensor([0.0], device=test_rep.device), torch.cumsum(test_sq, dim=0)])

    centers = (torch.arange(num_frames, device=test_rep.device) + 0.5) * frame_len
    starts = torch.clamp(centers - 0.5 * fs, min=0).to(torch.long)
    ends = torch.clamp(centers + 0.5 * fs, max=num_samples).to(torch.long)

    rms = (cum_sum[ends] - cum_sum[starts]) / torch.clamp(ends - starts, min=1)

    sort_idx = torch.argsort(local_p)
    sorted_p = local_p[sort_idx]
    sorted_rms = rms[sort_idx]

    cum_rms = torch.cumsum(sorted_rms, dim=0)
    cutoff = 0.5 * cum_rms[-1]
    matches = torch.where(cum_rms >= cutoff)[0]

    return float(sorted_p[matches[0]].item()) if len(matches) > 0 else 0.0


def calculate_auditory_quality_features(signals, fs=16000.0):
    """Natively computes similarity features for all separated channels."""
    true_target, target_dist, interf, artif = signals
    if true_target.dim() == 1:
        true_target = true_target.unsqueeze(1)
        target_dist = target_dist.unsqueeze(1)
        interf = interf.unsqueeze(1)
        artif = artif.unsqueeze(1)

    estimate = true_target + target_dist + interf + artif
    C = true_target.shape[1]

    ch_tar, ch_int, ch_art, ch_glo = [], [], [], []
    for c in range(C):
        mtest, fr = generate_auditory_internal_representation_torch(estimate[:, c], fs)

        mref_t, _ = generate_auditory_internal_representation_torch(true_target[:, c] + interf[:, c] + artif[:, c], fs)
        ch_tar.append(calculate_auditory_similarity_metric_torch(mref_t, mtest, fr))

        mref_i, _ = generate_auditory_internal_representation_torch(true_target[:, c] + target_dist[:, c] + artif[:, c], fs)
        ch_int.append(calculate_auditory_similarity_metric_torch(mref_i, mtest, fr))

        mref_a, _ = generate_auditory_internal_representation_torch(true_target[:, c] + target_dist[:, c] + interf[:, c], fs)
        ch_art.append(calculate_auditory_similarity_metric_torch(mref_a, mtest, fr))

        mref_g, _ = generate_auditory_internal_representation_torch(true_target[:, c], fs)
        ch_glo.append(calculate_auditory_similarity_metric_torch(mref_g, mtest, fr))

    return min(ch_tar), min(ch_int), min(ch_art), min(ch_glo)