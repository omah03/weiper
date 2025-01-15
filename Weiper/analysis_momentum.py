#!/usr/bin/env python3
"""
analysis_momentum.py

A script to:
 1) Load final-layer momentum logs (from momentum-based SGD)
 2) Convert them into a shape [N, D] array
 3) Load WeiPer random vectors
 4) Compare magnitudes & angles
 5) Save histograms

Usage:
  python analysis_momentum.py \
      --logs_path final_layer_momentum_logs.npy \
      --weiper_path weiper_noise_vectors.npy \
      --output_dir analysis_plots

Author: <Your Name>
Date: <Today’s Date>
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_momentum_logs(logs_path):
    """
    Loads the momentum logs from .npy, which is a list of dicts:
      [
        { "iter": (epoch, step),
          "layer": "fc.weight",
          "momentum_vec": np.array([...]) 
        },
        ...
      ]
    Returns: a Python list with these dicts.
    """
    logs = np.load(logs_path, allow_pickle=True)
    # logs is an array of Python dicts
    # safe to convert to list:
    if isinstance(logs, np.ndarray):
        logs = logs.tolist()
    print(f"[INFO] Loaded momentum logs from => {logs_path} | length={len(logs)}")
    return logs


def logs_to_2d_array(momentum_logs):
    """
    Convert the momentum logs into shape [N, D], 
    where each row is the final-layer momentum vector flattened.

    momentum_logs: list of dicts, each has "momentum_vec": np.array([...]) shape ~ [C,H,W] or [D]
    We'll flatten them.

    Returns: np.array of shape [N, D].
    """
    all_vecs = []
    for entry in momentum_logs:
        mom_vec = entry["momentum_vec"]  # shape e.g. [out_features, in_features]
        flat = mom_vec.flatten()
        all_vecs.append(flat)
    arr = np.stack(all_vecs, axis=0)
    print(f"[INFO] Created array from logs => shape={arr.shape}")
    return arr


def load_weiper_vectors(weiper_path):
    """
    Loads WeiPer random noise vectors from weiper_path (e.g. .npy).
    Expect shape [M, D].
    """
    data = np.load(weiper_path)
    print(f"[INFO] Loaded WeiPer vectors => shape={data.shape} from {weiper_path}")
    return data


def compare_magnitude_distributions(real_vectors, weiper_vectors, out_dir,
                                    real_label="Real Updates",
                                    weiper_label="WeiPer Noise",
                                    bins=50):
    """
    2.1 Magnitude Comparison
    real_vectors => shape [N, D]
    weiper_vectors => shape [M, D]

    Will produce a histogram for each dataset’s vector norms,
    save to out_dir/magnitude_comparison_hist.png
    """
    os.makedirs(out_dir, exist_ok=True)

    real_norms = np.linalg.norm(real_vectors, axis=1)
    weiper_norms = np.linalg.norm(weiper_vectors, axis=1)

    print(f"[MAGNITUDE] {real_label}: mean={real_norms.mean():.6f}, "
          f"min={real_norms.min():.6f}, max={real_norms.max():.6f}, count={len(real_norms)}")
    print(f"[MAGNITUDE] {weiper_label}: mean={weiper_norms.mean():.6f}, "
          f"min={weiper_norms.min():.6f}, max={weiper_norms.max():.6f}, count={len(weiper_norms)}")

    plt.figure(figsize=(8,5))
    plt.hist(real_norms, bins=bins, alpha=0.5, label=real_label, color='blue')
    plt.hist(weiper_norms, bins=bins, alpha=0.5, label=weiper_label, color='orange')
    plt.xlabel("Vector Norm", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Magnitude Distribution: Real vs. WeiPer", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    outpath = os.path.join(out_dir, "magnitude_comparison_hist.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved magnitude histogram => {outpath}")


def compare_angle_distributions(real_vectors, weiper_vectors, out_dir,
                                real_label="Real Updates",
                                weiper_label="WeiPer Noise",
                                bins=60):
    """
    2.2 Angle (Direction) Analysis
    We randomly pair real vs. WeiPer vectors (up to min(N,M) pairs)
    Then measure angles in degrees, histogram the results.

    real_vectors => shape [N, D]
    weiper_vectors => shape [M, D]
    """
    os.makedirs(out_dir, exist_ok=True)

    def to_unit(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True)+1e-12)

    real_unit = to_unit(real_vectors)
    weiper_unit = to_unit(weiper_vectors)

    N = len(real_unit)
    M = len(weiper_unit)
    use_len = min(N, M)

    idx_real = np.random.choice(N, use_len, replace=False)
    idx_weiper = np.random.choice(M, use_len, replace=False)

    real_sel = real_unit[idx_real]
    weiper_sel = weiper_unit[idx_weiper]

    dots = np.sum(real_sel*weiper_sel, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles_rad = np.arccos(dots)
    angles_deg = np.degrees(angles_rad)

    print(f"[ANGLE] Among {use_len} random pairs => "
          f"mean={angles_deg.mean():.2f} deg, std={angles_deg.std():.2f}, "
          f"min={angles_deg.min():.2f}, max={angles_deg.max():.2f}")

    plt.figure(figsize=(8,5))
    plt.hist(angles_deg, bins=bins, alpha=0.7, color='purple')
    plt.title("Angle Distribution: Real vs. WeiPer (Random Pairing)", fontsize=14)
    plt.xlabel("Angle (degrees)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)

    outpath = os.path.join(out_dir, "angle_comparison_hist.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved angle distribution => {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_path", type=str, required=True,
                        help="Path to final_layer_momentum_logs.npy")
    parser.add_argument("--weiper_path", type=str, required=True,
                        help="Path to weiper_noise_vectors.npy")
    parser.add_argument("--output_dir", type=str, default="analysis_plots",
                        help="Directory to save output plots")
    parser.add_argument("--last_k", type=int, default=100,
                        help="How many of the last momentum entries to use")
    args = parser.parse_args()

    # 1) Load momentum logs
    momentum_logs = load_momentum_logs(args.logs_path)

    # 2) Possibly only keep last_k
    if args.last_k < len(momentum_logs):
        momentum_logs = momentum_logs[-args.last_k:]
        print(f"[INFO] Using only the last {args.last_k} momentum entries.")

    # 3) Convert them to shape [N, D]
    real_vectors = logs_to_2d_array(momentum_logs)

    # 4) Load WeiPer random vectors from .npy
    weiper_vectors = load_weiper_vectors(args.weiper_path)

    # 5) Compare magnitude distributions
    compare_magnitude_distributions(real_vectors,
                                    weiper_vectors,
                                    args.output_dir,
                                    real_label="Momentum Updates",
                                    weiper_label="WeiPer Noise",
                                    bins=50)

    # 6) Compare angle distributions
    compare_angle_distributions(real_vectors,
                                weiper_vectors,
                                args.output_dir,
                                real_label="Momentum Updates",
                                weiper_label="WeiPer Noise",
                                bins=60)


if __name__ == "__main__":
    main()
