from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        default="logistic_linear",
        help="Sub-directory under alternating_minimization/ to aggregate.",
    )
    parser.add_argument("--datasets", nargs="+", default=["fashion_mnist", "cifar10"])
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8]
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base = (
        project_root / "logs" / "snn_weight_analysis" / "alternating_minimization" / args.variant
    )

    print(f"Variant: {args.variant}")
    summary: dict = {"variant": args.variant, "per_dataset": {}}

    for dataset in args.datasets:
        rows: list[dict] = []
        for seed in args.seeds:
            path = base / dataset / f"seed_{seed}" / "results.json"
            if not path.exists():
                print(f"  ! missing {path}")
                continue
            with open(path) as f:
                d = json.load(f)
            base_train = d["baseline"]["train"]
            base_test = d["baseline"]["test"]
            final_train = d["final"]["train"]
            final_test = d["final"]["test"]
            best_test = d["best_test"]
            # Derive train-best from history so pre-existing results.json (no
            # train_best key) still aggregate.
            hist = d.get("history", [])
            if hist:
                tb = max(hist, key=lambda h: h["train"])
                train_best_test = tb["test"]
            else:
                train_best_test = d.get("train_best", {}).get("test", final_test)
            rows.append(
                {
                    "seed": seed,
                    "base_train": base_train,
                    "base_test": base_test,
                    "final_train": final_train,
                    "final_test": final_test,
                    "train_best_test": train_best_test,
                    "best_test": best_test,
                    "dtrain_final": final_train - base_train,
                    "dtest_final": final_test - base_test,
                    "dtest_trainbest": train_best_test - base_test,
                    "dtest_best": best_test - base_test,
                    "n_iter": d["n_iterations"],
                }
            )

        if not rows:
            continue

        print(f"\n=== {dataset} ({len(rows)} seeds) ===")
        print(
            f"{'seed':>4}  "
            f"{'base_te':>8} {'tbest_te':>8} {'Δte_tbest':>9}  "
            f"{'fin_te':>8} {'Δte_fin':>8}  {'best_te':>8} {'Δte_oracle':>10}"
        )
        for r in rows:
            print(
                f"{r['seed']:>4}  "
                f"{r['base_test']:>8.4f} {r['train_best_test']:>8.4f} {r['dtest_trainbest']:>+9.4f}  "
                f"{r['final_test']:>8.4f} {r['dtest_final']:>+8.4f}  "
                f"{r['best_test']:>8.4f} {r['dtest_best']:>+10.4f}"
            )

        def stats(key: str) -> dict:
            vals = np.asarray([r[key] for r in rows])
            return {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
            }

        agg = {
            "n_seeds": len(rows),
            "base_train": stats("base_train"),
            "base_test": stats("base_test"),
            "final_train": stats("final_train"),
            "final_test": stats("final_test"),
            "train_best_test": stats("train_best_test"),
            "best_test": stats("best_test"),
            "dtrain_final": stats("dtrain_final"),
            "dtest_final": stats("dtest_final"),
            "dtest_trainbest": stats("dtest_trainbest"),
            "dtest_best": stats("dtest_best"),
        }
        print(
            f"\n  mean Δtest (train-best) [HEADLINE]: "
            f"{agg['dtest_trainbest']['mean']:+.4f} ± {agg['dtest_trainbest']['std']:.4f}  (n={len(rows)})"
        )
        print(
            f"  mean Δtest (final):                 {agg['dtest_final']['mean']:+.4f} ± {agg['dtest_final']['std']:.4f}"
        )
        print(
            f"  mean Δtest (best-test) [oracle]:    {agg['dtest_best']['mean']:+.4f} ± {agg['dtest_best']['std']:.4f}"
        )

        summary["per_dataset"][dataset] = {"per_seed": rows, "aggregate": agg}

    out = base / "summary_across_seeds.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")

    # Plot: per-seed best-test Δ as bar chart, both datasets
    fig, axes = plt.subplots(
        1,
        len([d for d in args.datasets if d in summary["per_dataset"]]),
        figsize=(10, 4),
        squeeze=False,
    )
    for ax, dataset in zip(
        axes[0],
        [d for d in args.datasets if d in summary["per_dataset"]],
    ):
        rows = summary["per_dataset"][dataset]["per_seed"]
        agg = summary["per_dataset"][dataset]["aggregate"]
        seeds = [r["seed"] for r in rows]
        d_trainbest = [r["dtest_trainbest"] * 100 for r in rows]
        d_oracle = [r["dtest_best"] * 100 for r in rows]
        x = np.arange(len(seeds))
        w = 0.35
        ax.bar(x - w / 2, d_trainbest, w, label="train-best (headline)", color="C0")
        ax.bar(x + w / 2, d_oracle, w, label="best-test (oracle)", color="C2", alpha=0.5)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(
            agg["dtest_trainbest"]["mean"] * 100,
            color="C0",
            ls="--",
            alpha=0.6,
            label=f"mean train-best {agg['dtest_trainbest']['mean'] * 100:+.2f} pp",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("seed")
        ax.set_ylabel("Δtest (pp)")
        ax.set_title(
            f"{dataset}  Δtest (train-best) = "
            f"{agg['dtest_trainbest']['mean'] * 100:+.2f} ± {agg['dtest_trainbest']['std'] * 100:.2f} pp"
        )
        ax.legend(fontsize=8)
    fig.suptitle(f"Alternating minimization: {args.variant}")
    fig.tight_layout()
    fig.savefig(base / "summary_across_seeds.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {base}/summary_across_seeds.png")


if __name__ == "__main__":
    main()
