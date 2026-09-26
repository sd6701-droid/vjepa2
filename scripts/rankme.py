"""RankMe (Garrido et al., 2023) for every checkpoint of a V-JEPA 2 / 2.1 pretraining run.

RankMe = exp(entropy of the normalized singular values of the embedding matrix).
Higher = features use more dimensions; a drop towards 1 means collapse.
Max possible value is min(num_samples, embed_dim) (384 for ViT-S).

Usage (from the repo root, vjepa2 env, on a GPU node):
    python scripts/rankme.py <run_dir> --data <val.csv> --every 10

Reads <run_dir>/params-pretrain.yaml to rebuild the encoder (app: vjepa or
vjepa_2_1), evaluates every <run_dir>/eN.pt (V-JEPA 2) or eN.pth.tar (V-JEPA 2.1)
checkpoint -- only N % --every == 0 if given, plus latest with --include-latest --
on the same fixed set of clips, and appends results to <run_dir>/rankme.csv. Checkpoints
already in the csv are skipped, so it can be re-run while training continues.
With --wandb, each result is also logged to a separate "<run>-rankme" wandb run
(same project, grouped with the training run); re-runs append to that same run.

Two numbers per checkpoint:
    rankme_pooled : clip embeddings (mean over all tokens)  -> global features
    rankme_tokens : individual patch tokens                  -> dense features
"""

import argparse
import csv
import glob
import hashlib
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets.video_dataset import VideoDataset  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def rankme(z, eps=1e-7):
    s = torch.linalg.svdvals(z.double())
    p = s / (s.sum() + eps)
    return float(torch.exp(-(p * torch.log(p + eps)).sum()))


class CenterCropUint8:
    """[T,H,W,3] uint8 -> [3,T,S,S] uint8: resize short side to S, center crop."""

    def __init__(self, size):
        self.size = size

    def __call__(self, buffer):
        x = torch.from_numpy(np.asarray(buffer)).permute(0, 3, 1, 2).float()  # T,3,H,W
        H, W = x.shape[-2:]
        scale = self.size / min(H, W)
        h, w = max(self.size, round(H * scale)), max(self.size, round(W * scale))
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        top, left = (h - self.size) // 2, (w - self.size) // 2
        x = x[:, :, top : top + self.size, left : left + self.size]
        return x.round().clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3)  # 3,T,S,S


def load_clips(p, data_csv, num_samples, seed, num_workers, cache):
    if cache and os.path.exists(cache):
        print(f"loading cached clips from {cache}")
        return torch.load(cache)

    d = p["data"]
    fpc = max(d["dataset_fpcs"])
    dataset = VideoDataset(
        data_paths=[data_csv],
        frames_per_clip=fpc,
        dataset_fpcs=[fpc],
        fps=d.get("fps"),
        frame_step=None if d.get("fps") else d.get("frame_step", 4),
        num_clips=1,
        random_clip_sampling=False,  # deterministic clip position
        transform=CenterCropUint8(d["crop_size"]),
    )
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, idx.tolist()),
        batch_size=16,
        num_workers=num_workers,
        collate_fn=lambda b: torch.stack([s[0][0] for s in b]),
    )
    t0 = time.time()
    clips = torch.cat([b for b in loader])  # N,3,T,S,S uint8
    print(f"decoded {len(clips)} clips {tuple(clips.shape[1:])} in {time.time() - t0:.0f}s")
    if cache:
        torch.save(clips, cache)
        print(f"cached clips to {cache}")
    return clips


CKPT_EXT = {"vjepa": ".pt", "vjepa_2_1": ".pth.tar"}


def build_encoder(p, device):
    m, d = p["model"], p["data"]
    fpcs = d["dataset_fpcs"]
    if p["app"] == "vjepa":
        from app.vjepa.utils import init_video_model

        encoder, _ = init_video_model(
            device=device,
            patch_size=d["patch_size"],
            max_num_frames=max(fpcs),
            tubelet_size=d["tubelet_size"],
            model_name=m["model_name"],
            crop_size=d["crop_size"],
            pred_depth=m["pred_depth"],
            pred_num_heads=m.get("pred_num_heads"),
            pred_embed_dim=m["pred_embed_dim"],
            uniform_power=m.get("uniform_power", False),
            use_mask_tokens=m.get("use_mask_tokens", False),
            num_mask_tokens=len(p["mask"]) * len(fpcs),
            zero_init_mask_tokens=m.get("zero_init_mask_tokens", True),
            use_sdpa=p["meta"].get("use_sdpa", False),
            use_rope=m.get("use_rope", False),
            use_silu=m.get("use_silu", False),
            use_pred_silu=m.get("use_pred_silu", False),
            wide_silu=m.get("wide_silu", True),
            use_activation_checkpointing=False,
        )
        return encoder.eval()

    from app.vjepa_2_1.utils import init_video_model

    encoder, _ = init_video_model(
        device=device,
        patch_size=d["patch_size"],
        max_num_frames=max(fpcs),
        tubelet_size=d["tubelet_size"],
        model_name=m["model_name"],
        crop_size=d["crop_size"],
        pred_depth=m["pred_depth"],
        pred_num_heads=m.get("pred_num_heads"),
        pred_embed_dim=m["pred_embed_dim"],
        uniform_power=m.get("uniform_power", False),
        use_mask_tokens=m.get("use_mask_tokens", False),
        num_mask_tokens=len(p["mask"]) * len(fpcs),
        zero_init_mask_tokens=m.get("zero_init_mask_tokens", True),
        use_sdpa=p["meta"].get("use_sdpa", False),
        use_rope=m.get("use_rope", False),
        is_causal=m.get("is_causal", False),
        pred_is_causal=m.get("pred_is_causal", False),
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
    return encoder.eval()


@torch.no_grad()
def embed(encoder, clips, batch_size, tokens_per_clip, device, seed):
    g = torch.Generator().manual_seed(seed)
    pooled, tokens = [], []
    for i in range(0, len(clips), batch_size):
        x = clips[i : i + batch_size].float().div_(255)
        x = ((x - MEAN) / STD).to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            h = encoder([x])[0]  # B, N, D (last layer, layer-normed)
        h = h.float().cpu()
        pooled.append(h.mean(dim=1))
        pick = torch.randint(h.shape[1], (h.shape[0], tokens_per_clip), generator=g)
        tokens.append(torch.gather(h, 1, pick[..., None].expand(-1, -1, h.shape[2])).flatten(0, 1))
    return torch.cat(pooled), torch.cat(tokens)


def file_index(path):
    """N in eN.pt / eN.pth.tar (the checkpoint holds the state after N+1 epochs)."""
    m = re.fullmatch(r"e(\d+)\.(pt|pth\.tar)", os.path.basename(path))
    return int(m.group(1)) if m else None


def init_wandb(args, p, data_csv):
    import wandb

    run_name = os.path.basename(os.path.abspath(args.run_dir))
    # same settings -> same wandb id, so re-runs append points to one run
    key = f"{os.path.abspath(args.run_dir)}|{data_csv}|{args.num_samples}|{args.seed}|{args.which}"
    run = wandb.init(
        project=args.wandb_project or p.get("meta", {}).get("wandb_project", "vjepa2"),
        name=f"{run_name}-rankme",
        group=run_name,
        job_type="rankme",
        id="rankme-" + hashlib.md5(key.encode()).hexdigest()[:12],
        resume="allow",
        dir=args.run_dir,
        config={"run_dir": os.path.abspath(args.run_dir), "app": p["app"], "data": data_csv,
                "num_samples": args.num_samples, "tokens_per_clip": args.tokens_per_clip,
                "which": args.which, "seed": args.seed, "model": p["model"]["model_name"]},
    )
    run.define_metric("epoch")
    run.define_metric("rankme/*", step_metric="epoch")
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--data", default=None,
                    help="csv of clips to embed (default: first training csv; a val csv is better)")
    ap.add_argument("--num-samples", type=int, default=2048)
    ap.add_argument("--tokens-per-clip", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--which", default="target_encoder", choices=["target_encoder", "encoder"])
    ap.add_argument("--every", type=int, default=1, help="only eN checkpoints with N %% every == 0")
    ap.add_argument("--include-latest", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="recompute checkpoints already in the csv")
    ap.add_argument("--cache", default=None, help="file to save/load the decoded clips (uint8)")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true", help="also log results to wandb")
    ap.add_argument("--wandb-project", default=None, help="default: meta.wandb_project of the run")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "params-pretrain.yaml")) as f:
        p = yaml.safe_load(f)
    if p.get("app") not in CKPT_EXT:
        sys.exit(f"expected app: vjepa or vjepa_2_1, got {p.get('app')}")
    ext = CKPT_EXT[p["app"]]
    latest = "latest" + ext

    ckpts = [c for c in glob.glob(os.path.join(args.run_dir, "e*" + ext)) if file_index(c) is not None]
    ckpts = sorted((c for c in ckpts if file_index(c) % args.every == 0), key=file_index)
    if args.include_latest and os.path.exists(os.path.join(args.run_dir, latest)):
        ckpts.append(os.path.join(args.run_dir, latest))

    out_csv = os.path.join(args.run_dir, "rankme.csv")
    done = set()
    if os.path.exists(out_csv) and not args.overwrite:
        with open(out_csv) as f:
            done = {r["checkpoint"] for r in csv.DictReader(f) if r["checkpoint"] != latest}
    todo = [c for c in ckpts if os.path.basename(c) not in done]
    print(f"app={p['app']}: {len(ckpts)} checkpoints selected in {args.run_dir}, {len(todo)} to evaluate")
    if not todo:
        return

    data_csv = args.data or p["data"]["datasets"][0]
    print(f"clips from {data_csv}")
    clips = load_clips(p, data_csv, args.num_samples, args.seed, args.num_workers, args.cache)
    encoder = build_encoder(p, args.device)
    wandb_run = init_wandb(args, p, data_csv) if args.wandb else None

    fields = ["checkpoint", "epoch", "rankme_pooled", "rankme_tokens", "embed_dim",
              "num_clips", "num_tokens", "train_loss", "which"]
    new_file = args.overwrite or not os.path.exists(out_csv)
    with open(out_csv, "w" if new_file else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for c in todo:
            ckpt = torch.load(c, map_location="cpu", weights_only=False)
            sd = {k.removeprefix("module."): v for k, v in ckpt[args.which].items()}
            encoder.load_state_dict(sd, strict=True)
            pooled, tokens = embed(encoder, clips, args.batch_size, args.tokens_per_clip,
                                   args.device, args.seed)
            row = {
                "checkpoint": os.path.basename(c),
                "epoch": ckpt.get("epoch"),
                "rankme_pooled": f"{rankme(pooled):.2f}",
                "rankme_tokens": f"{rankme(tokens):.2f}",
                "embed_dim": pooled.shape[1],
                "num_clips": pooled.shape[0],
                "num_tokens": tokens.shape[0],
                "train_loss": f"{ckpt.get('loss', float('nan')):.4f}",
                "which": args.which,
            }
            w.writerow(row)
            f.flush()
            print(f"{row['checkpoint']:>16s}  epoch {row['epoch']!s:>4}  "
                  f"rankme pooled {row['rankme_pooled']:>7s} / tokens {row['rankme_tokens']:>7s}  "
                  f"(max {row['embed_dim']})  loss {row['train_loss']}")
            if wandb_run is not None:
                wandb_run.log({
                    "epoch": row["epoch"],
                    "rankme/pooled": float(row["rankme_pooled"]),
                    "rankme/tokens": float(row["rankme_tokens"]),
                    "rankme/train_loss": float(row["train_loss"]),
                })
            del ckpt
    print(f"wrote {out_csv}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
