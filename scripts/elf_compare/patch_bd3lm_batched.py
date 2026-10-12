"""Patch the bd3lms checkout's diffusion.py so its samplers also run at batch > 1 (throughput benchmark only).

The official code assumes batch 1 in three places: the first-hitting time uses a scalar `.item()` masked
count, the first-hitting update draws one masked index with torch.randint(0, <tensor>), and nucleus
sampling sorts the whole [B, L, V] float64 tensor at once (24.5 GB at B=32). Row-wise versions below are
mathematically the same procedure applied per row; batch-1 behaviour is untouched. Original kept as
diffusion.py.orig. Run inside the bd3lms directory:  python patch_bd3lm_batched.py
"""
import os
import shutil

p = "diffusion.py"
if not os.path.exists(p + ".orig"):
    shutil.copy(p, p + ".orig")
s = open(p, encoding="utf-8").read()

old_fh_time = """          u = np.random.rand()
          num_masked = (x_accum[:, fwd_idx] == self.mask_index).sum(-1).item()
          t *= u**(1 / num_masked)
"""
new_fh_time = """          u = np.random.rand()
          nm = (x_accum[:, fwd_idx] == self.mask_index).sum(-1)
          if nm.numel() == 1:
            num_masked = nm.item()
            t *= u**(1 / num_masked)
          else:  # batched benchmark: per-row first-hitting time; rows without masks are no-ops
            t = t * u**(1 / nm.clamp_min(1).float())
"""
old_idx = """      num_masked = (x[:, -self.block_size:] == self.mask_index).sum(-1)
      ind = torch.randint(0, num_masked, (x_block.shape[0],))
      ind = (x[:, -self.block_size:] == self.mask_index).nonzero()[ind, 1]
      mask = (torch.arange(self.block_size, device=x.device) == ind[:, None]).to(x_block.dtype)
"""
new_idx = """      masked = (x[:, -self.block_size:] == self.mask_index)
      num_masked = masked.sum(-1)
      if num_masked.numel() == 1:
        ind = torch.randint(0, num_masked, (x_block.shape[0],))
        ind = masked.nonzero()[ind, 1]
      else:  # batched benchmark: per-row uniform choice among the masked positions of that row
        kth = (torch.rand(x.shape[0], device=x.device) * num_masked.clamp_min(1).float()).long()
        cum = masked.long().cumsum(-1) - 1
        pos = ((cum == kth[:, None]) & masked).float().argmax(-1)
        ind = torch.where(num_masked > 0, pos, torch.full_like(pos, -1))
      mask = (torch.arange(self.block_size, device=x.device) == ind[:, None]).to(x_block.dtype)
"""
old_nuc = """    p_x0_ = p_x0[:, -self.block_size:].clone()
    sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    nucleus_mask = cum_probs <= p
    nucleus_mask[..., 0] = 1
    sorted_probs = sorted_probs * nucleus_mask
    p_x0_.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
    p_x0_ /= p_x0_.sum(-1, keepdim=True)
    p_x0[:, -self.block_size:] = p_x0_
    return p_x0
"""
new_nuc = """    p_x0_ = p_x0[:, -self.block_size:].clone()
    for b in range(p_x0_.shape[0]):  # row-wise: bounds the sort memory at large batch, same result
      row = p_x0_[b:b + 1]
      sorted_probs, sorted_indices = row.sort(dim=-1, descending=True)
      cum_probs = sorted_probs.cumsum(dim=-1)
      nucleus_mask = cum_probs <= p
      nucleus_mask[..., 0] = 1
      sorted_probs = sorted_probs * nucleus_mask
      row.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
      row /= row.sum(-1, keepdim=True)
    p_x0[:, -self.block_size:] = p_x0_
    return p_x0
"""
n = 0
KEYS = {"first-hitting time": "nm.clamp_min(1).float()", "first-hitting index": "num_masked.clamp_min(1).float()",
        "nucleus": "for b in range(p_x0_.shape[0])"}
for old, new, tag in [(old_fh_time, new_fh_time, "first-hitting time"), (old_idx, new_idx, "first-hitting index"),
                      (old_nuc, new_nuc, "nucleus")]:
    if KEYS[tag] in s:
        print(f"{tag}: already patched")
    elif old in s:
        s = s.replace(old, new)
        n += 1
        print(f"{tag}: patched")
    else:
        raise SystemExit(f"{tag}: anchor not found")
open(p, "w", encoding="utf-8").write(s)
print(f"done ({n} new patches)")
