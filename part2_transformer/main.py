"""
EEG Transformer Sprint — entry point.
Run:  python main.py

This script:
  1. loads the cross-subject split (train subjects 1-8, test 9-10),
  2. trains the two baselines (EEGNet, vanilla transformer),
  3. trains YOUR model across a few seeds and reports best + worst,
  4. saves one figure (figure.png).

>>> YOUR JOB <<<
Implement ONE inductive bias in `YourModel` below (or replace it entirely).
State your hypothesis in the README. Do not quietly edit the vanilla
transformer in models.py — the contrast is the point.
"""
import argparse
import numpy as np
import torch
import json 
import torch.nn.functional as Fn

from data import load_data, get_channel_positions
from models import EEGNet, VanillaEEGTransformer, count_params
from train import train_model, set_seed


# =========================================================================== #
#  YOUR MODEL  — the only part you must write.
#
#  The starter version below is the vanilla transformer with learned TEMPORAL
#  PATCH tokens instead of per-timestep tokens: one concrete, legal example of
#  "inject one inductive bias" (fewer tokens -> a smaller O(T^2) surface to
#  overfit on 8 subjects). Replace its guts with your own idea. Some directions:
#    - structured positional encoding from electrode geometry (get_channel_positions)
#      and/or frequency band,
#    - an attention bias toward scalp neighbourhoods or band-limited windows,
#    - channel (spatial) tokens instead of / in addition to time tokens,
#    - regularisation aimed specifically at the 8-subject data wall.
# =========================================================================== #
import torch.nn as nn


class PatchTransformer(nn.Module):
    def __init__(self, n_chan, n_time, n_classes=2,
                 patch=32, d_model=64, n_heads=4, depth=3, p=0.3):
        super().__init__()
        # Learned temporal patches: a Conv1d with stride == kernel tokenises
        # the signal into n_time//patch tokens, each summarising `patch` samples
        # across all electrodes. This is the ONE bias in the starter example.
        self.tokenizer = nn.Conv1d(n_chan, d_model, kernel_size=patch,
                                   stride=patch)
        n_tokens = n_time // patch
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=p, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, return_embed=False):   # x: (B, C, T)
        tok = self.tokenizer(x).transpose(1, 2)  # (B, n_tokens, d)
        cls = self.cls.expand(x.size(0), -1, -1)
        z = torch.cat([cls, tok], dim=1) + self.pos
        z = self.encoder(z)
        embed = self.norm(z[:, 0])
        logits = self.head(embed)
        return (logits, embed) if return_embed else logits


# =========================================================================== #


def run_across_seeds(build_fn, Xtr, ytr, Xte, yte, seeds, device, **kw):
    bests, finals = [], []
    for s in seeds:
        set_seed(s)
        model = build_fn()
        best, final = train_model(model, Xtr, ytr, Xte, yte, device=device, **kw)
        bests.append(best); finals.append(final)
        print(f"  seed {s}: FINAL-epoch test acc {final:.3f}   (peak {best:.3f})")
    return np.array(bests), np.array(finals)


def summarize(name, bests, finals):
    print(f"{name}: FINAL mean {finals.mean():.3f} std {finals.std():.3f} "
          f"best {finals.max():.3f} worst {finals.min():.3f} | peak-epoch mean {bests.mean():.3f}")


def save_figure(model, Xte, yte, device, path="figure.png"):
    """t-SNE of the CLS embeddings on the test subjects, coloured by class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(Xte).to(device), return_embed=True)
        emb = out[1].cpu().numpy() if isinstance(out, tuple) else None
    if emb is None:
        print("  (your model did not return an embedding; skipping figure)")
        return
    n = len(emb)
    emb2 = TSNE(n_components=2, perplexity=min(30, max(5, n // 4)),
                init="pca", random_state=0).fit_transform(emb)
    plt.figure(figsize=(5, 4))
    for c, name in [(0, "fists"), (1, "feet")]:
        m = yte == c
        plt.scatter(emb2[m, 0], emb2[m, 1], s=18, alpha=0.7, label=name)
    plt.legend(); plt.title("Test-subject embeddings (your model)")
    plt.tight_layout(); plt.savefig(path, dpi=130)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ablate", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print("\nloading data...")
    Xtr, ytr, Xte, yte = load_data()
    C, T = Xtr.shape[1], Xtr.shape[2]
    print(f"train {Xtr.shape}  test {Xte.shape}  "
          f"(class balance train {ytr.mean():.2f} / test {yte.mean():.2f})")

    builders = {
        "EEGNet": lambda: EEGNet(C, T),
        "Vanilla transformer": lambda: VanillaEEGTransformer(C, T),
        "Patch transformer (starter)": lambda: PatchTransformer(C, T),
    }
    # YourModel rows are added in step 4

    open("results_part2.jsonl", "w").close()
    for name, build in builders.items():
        print(f"\n=== {name} ===")
        set_seed(0)
        print(f"params: {count_params(build()):,}")
        bests, finals = run_across_seeds(build, Xtr, ytr, Xte, yte, args.seeds,
                                         device, epochs=args.epochs)
        summarize(name, bests, finals)
        with open("results_part2.jsonl", "a") as f:
            for s, b, fi in zip(args.seeds, bests, finals):
                f.write(json.dumps(dict(model=name, seed=s,
                                        final_acc=float(fi), peak_acc=float(b))) + "\n")

    print("\nsaving figure...")
    fig_key = "YourModel" if "YourModel" in builders else "Patch transformer (starter)"
    set_seed(0)
    fm = builders[fig_key]()
    train_model(fm, Xtr, ytr, Xte, yte, epochs=args.epochs, device=device)
    save_figure(fm, Xte, yte, device)
    print("\ndone. Report FINAL-epoch numbers, best AND worst seeds, no cherry-picking.")
if __name__ == "__main__":
    main()
