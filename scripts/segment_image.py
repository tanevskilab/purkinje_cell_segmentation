"""Apply a trained CellposeSAM model to an H&E image and save the results.

Usage:
    python segment_image.py <model_path> <image_path>
    python segment_image.py cellpose_training_crops/models/cp4_20260907_102211 \\
        cellpose_test_crops/some_crop.tif --output-dir results/
"""
import argparse
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from cellpose import models, plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path",
                         help="Path to a trained model, e.g. under cellpose_training_crops/models/")
    parser.add_argument("image_path",
                         help="Path to the H&E image to segment (.tif/.png/.jpg)")
    parser.add_argument("--output-dir", default=None,
                         help="Where to save results. Defaults to the image's own folder.")
    parser.add_argument("--diameter", type=float, default=None,
                         help="Expected cell diameter in pixels. Defaults to auto-estimate.")
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--no-gpu", action="store_true",
                         help="Force CPU even if a GPU/MPS device is available.")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir) if args.output_dir else image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    model = models.CellposeModel(gpu=not args.no_gpu, pretrained_model=str(model_path))

    print(f"Loading image: {image_path}")
    img = tifffile.imread(image_path)

    print("Running segmentation...")
    masks, flows, styles = model.eval(
        img,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        normalize=True,
    )
    n_cells = int(masks.max())
    print(f"Detected {n_cells} cells")

    stem = image_path.stem

    masks_path = output_dir / f"{stem}_masks.tif"
    tifffile.imwrite(masks_path, masks.astype(np.uint16))
    print(f"Saved masks to {masks_path}")

    overlay = plot.mask_overlay(img, masks)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title(image_path.name, fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"{n_cells} cells", fontsize=9)
    axes[1].axis("off")
    plt.tight_layout()

    plot_path = output_dir / f"{stem}_overlay.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved overlay plot to {plot_path}")


if __name__ == "__main__":
    main()
