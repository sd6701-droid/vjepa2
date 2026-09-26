"""Sanity-check a V-JEPA 2.1 pretraining checkpoint without touching the run.

Usage (from the repo root, vjepa2 env):
    python scripts/check_ckpt.py <run_dir>/e49.pth.tar            # inspect only
    python scripts/check_ckpt.py <run_dir>/e49.pth.tar --forward  # + build the
        encoder from <run_dir>/params-pretrain.yaml and push a random clip through it

Reports: top-level keys, epoch/loss metadata, per-module tensor stats,
NaN/Inf, the patch-embed shape (tubelet size), encoder-vs-target-encoder
EMA drift, optimizer step count, and the loss curve from log_r0.csv.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import torch


def tensor_stats(sd, name):
    n_params = 0
    n_nan = n_inf = 0
    n_zero_tensors = 0
    per_prefix = defaultdict(float)
    for k, v in sd.items():
        if not torch.is_tensor(v) or not v.is_floating_point():
            continue
        n_params += v.numel()
        n_nan += torch.isnan(v).sum().item()
        n_inf += torch.isinf(v).sum().item()
        if v.numel() > 0 and v.abs().max().item() == 0:
            n_zero_tensors += 1
        per_prefix[k.split(".")[0] if "." not in k else ".".join(k.split(".")[:2])] += v.float().norm().item() ** 2
    print(f"\n[{name}] {len(sd)} tensors, {n_params/1e6:.2f}M params, "
          f"NaN={n_nan} Inf={n_inf} all-zero-tensors={n_zero_tensors}")
    if n_nan or n_inf:
        print("  !!! NaN/Inf present -- checkpoint is corrupt / training diverged")
    return n_params


def find_patch_embed(sd):
    for k, v in sd.items():
        if "patch_embed.proj.weight" in k and "patch_embed_img" not in k:
            return k, tuple(v.shape)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--forward", action="store_true",
                    help="build the encoder from params-pretrain.yaml and run a random clip")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = os.path.dirname(os.path.abspath(args.ckpt))
    print(f"checkpoint : {args.ckpt}")
    print(f"size       : {os.path.getsize(args.ckpt)/1e9:.2f} GB")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # ---- top-level keys -------------------------------------------------------
    print("\n== top-level keys ==")
    for k, v in ckpt.items():
        if isinstance(v, dict):
            print(f"  {k:16s} dict  ({len(v)} entries)")
        elif torch.is_tensor(v):
            print(f"  {k:16s} tensor {tuple(v.shape)}")
        else:
            print(f"  {k:16s} {type(v).__name__} = {v}")

    epoch = ckpt.get("epoch")
    print(f"\nepoch stored = {epoch}  (file eN.pth.tar is saved at end of epoch N+1, i.e. epoch key = N+1)")
    print(f"loss (avg over last epoch) = {ckpt.get('loss')}")

    # ---- weights --------------------------------------------------------------
    enc, tgt, pred = ckpt["encoder"], ckpt["target_encoder"], ckpt["predictor"]
    tensor_stats(enc, "encoder")
    tensor_stats(tgt, "target_encoder")
    tensor_stats(pred, "predictor")

    k, shape = find_patch_embed(enc)
    print(f"\npatch embed : {k} -> {shape}")
    if shape is not None and len(shape) == 5:
        print(f"  embed_dim={shape[0]} in_chans={shape[1]} tubelet={shape[2]} patch={shape[3]}x{shape[4]}")

    print("\nencoder first / last keys:")
    ks = list(enc.keys())
    for kk in ks[:4] + ["..."] + ks[-4:]:
        print(f"  {kk}" if kk == "..." else f"  {kk:60s} {tuple(enc[kk].shape)}")

    # ---- EMA drift: target should track the encoder but not equal it ---------
    print("\n== encoder vs target_encoder (EMA) ==")
    diffs = []
    missing = 0
    for kk, v in enc.items():
        if kk not in tgt or not v.is_floating_point():
            missing += 1
            continue
        d = (v.float() - tgt[kk].float()).norm().item()
        n = v.float().norm().item() + 1e-12
        diffs.append((d / n, kk))
    diffs.sort(reverse=True)
    rel = [d for d, _ in diffs]
    print(f"  keys compared={len(diffs)} missing_in_target={missing}")
    print(f"  mean rel-diff={sum(rel)/len(rel):.4e}  max rel-diff={rel[0]:.4e} ({diffs[0][1]})")
    if rel[0] == 0:
        print("  !!! target == encoder exactly: EMA never updated?")
    elif sum(rel) / len(rel) > 0.5:
        print("  !!! target very far from encoder: EMA momentum or loading is off")
    else:
        print("  ok: target is close to but distinct from the encoder (expected for EMA)")

    # ---- optimizer ------------------------------------------------------------
    opt = ckpt.get("opt")
    if opt:
        steps = sorted({int(s["step"]) for s in opt["state"].values() if "step" in s})
        print(f"\n== optimizer == groups={len(opt['param_groups'])} "
              f"tracked params={len(opt['state'])} step(s)={steps[:3]}{'...' if len(steps) > 3 else ''}")
        for i, g in enumerate(opt["param_groups"]):
            print(f"  group{i}: lr={g.get('lr'):.3e} wd={g.get('weight_decay')} n={len(g['params'])}")
        if epoch is not None and steps:
            print(f"  expected steps = epoch*ipe = {epoch}*300 = {epoch*300}; stored = {steps[-1]}")

    # ---- loss curve from the csv log -----------------------------------------
    log = os.path.join(run_dir, "log_r0.csv")
    if os.path.exists(log):
        per_epoch = defaultdict(list)
        with open(log) as f:
            r = csv.reader(f)
            header = next(r)
            for row in r:
                try:
                    per_epoch[int(row[0])].append(float(row[2]))
                except (ValueError, IndexError):
                    pass
        eps = sorted(per_epoch)
        print(f"\n== {log} == epochs logged: {eps[0]}..{eps[-1]}")
        show = sorted(set(eps[:3] + [e for e in eps if e % 10 == 0] + eps[-2:]))
        for e in show:
            ls = per_epoch[e]
            print(f"  epoch {e:4d}: mean loss {sum(ls)/len(ls):.4f}  (n={len(ls)})")
        if epoch in per_epoch:
            print(f"  ckpt['loss']={ckpt.get('loss'):.4f} vs csv epoch {epoch} mean="
                  f"{sum(per_epoch[epoch])/len(per_epoch[epoch]):.4f}")
    else:
        print(f"\n(no {log})")

    # ---- forward pass ----------------------------------------------------------
    if args.forward:
        import yaml
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from app.vjepa_2_1.utils import init_video_model

        with open(os.path.join(run_dir, "params-pretrain.yaml")) as f:
            p = yaml.safe_load(f)
        m, d = p["model"], p["data"]
        fpcs = d["dataset_fpcs"]
        encoder, predictor = init_video_model(
            device=args.device,
            patch_size=d["patch_size"], max_num_frames=max(fpcs),
            tubelet_size=d["tubelet_size"], model_name=m["model_name"],
            crop_size=d["crop_size"], pred_depth=m["pred_depth"],
            pred_num_heads=m.get("pred_num_heads"), pred_embed_dim=m["pred_embed_dim"],
            uniform_power=m.get("uniform_power", False),
            use_mask_tokens=m.get("use_mask_tokens", False),
            num_mask_tokens=len(p["mask"]) * len(fpcs),
            zero_init_mask_tokens=m.get("zero_init_mask_tokens", True),
            use_sdpa=p["meta"].get("use_sdpa", False), use_rope=m.get("use_rope", False),
            is_causal=m.get("is_causal", False), pred_is_causal=m.get("pred_is_causal", False),
            use_activation_checkpointing=False,
            return_all_tokens=p["loss"].get("predict_all", False),
            chop_last_n_tokens=p["loss"].get("shift_by_n", 0),
            img_temporal_dim_size=m.get("img_temporal_dim_size"),
            n_registers=m.get("n_registers", 0),
            n_registers_predictor=m.get("n_registers_predictor", 0),
            has_cls_first=m.get("has_cls_first", False),
            interpolate_rope=m.get("interpolate_rope", False),
            modality_embedding=m.get("modality_embedding", False),
        )
        msg = encoder.load_state_dict(tgt, strict=True)
        print(f"\n== forward == target_encoder loaded strictly: {msg}")
        encoder.eval()
        x = torch.randn(1, 3, max(fpcs), d["crop_size"], d["crop_size"], device=args.device)
        with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
            out = encoder([x])[0]
        print(f"  input {tuple(x.shape)} -> output {tuple(out.shape)}")
        exp_tokens = (max(fpcs) // d["tubelet_size"]) * (d["crop_size"] // d["patch_size"]) ** 2
        print(f"  expected tokens = {exp_tokens}; finite={torch.isfinite(out).all().item()} "
              f"mean={out.float().mean():.4f} std={out.float().std():.4f}")
        # a trained encoder gives token features that are not all identical
        tok_std = out.float().std(dim=1).mean().item()
        print(f"  per-dim std across tokens = {tok_std:.4f} (near 0 would mean collapsed features)")


if __name__ == "__main__":
    main()
