"""Real vs. fake average power-spectrum plots, for the error-analysis doc and demo video."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.frequency.features import _radial_profile, _to_gray  # noqa: SLF001 — internal reuse is intentional here
from src.frequency.train_svm import list_images


def average_radial_spectrum(paths: list[str], n_bins: int = 32) -> np.ndarray:
    profiles = []
    for path in paths:
        gray = _to_gray(Image.open(path).convert("RGB"))
        spectrum = np.fft.fftshift(np.fft.fft2(gray * np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))))
        power = np.log1p(np.abs(spectrum) ** 2)
        profiles.append(_radial_profile(power, n_bins))
    return np.mean(profiles, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Plot mean real vs. fake radial power spectrum.")
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--fake-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--out", default="outputs/figures/spectrum_real_vs_fake.png")
    args = parser.parse_args()

    real_paths = list_images(args.real_dir)[: args.n_samples]
    fake_paths = list_images(args.fake_dir)[: args.n_samples]

    real_profile = average_radial_spectrum(real_paths)
    fake_profile = average_radial_spectrum(fake_paths)

    plt.figure(figsize=(6, 4))
    plt.plot(real_profile, label="real", marker="o")
    plt.plot(fake_profile, label="fake (AIGC)", marker="o")
    plt.xlabel("radial frequency bin (low -> high)")
    plt.ylabel("mean log power")
    plt.title("Mean radially-averaged power spectrum: real vs. AIGC")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
