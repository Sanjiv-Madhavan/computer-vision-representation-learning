"""First experiment: frozen CLIP image embeddings for three user-provided photos.

Run from the project root:
    python -m probe1.experiment_01
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without requiring a GUI backend.
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from probe1.similarity import cosine_similarity_matrix, husky_wolf_margin, normalized_embeddings

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("husky", "wolf", "retriever")
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_images_present(image_dir: Path) -> None:
    missing = [f"{name}.jpg" for name in NAMES if not (image_dir / f"{name}.jpg").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing image(s) in {image_dir}: {', '.join(missing)}. "
            "See README.md for the required photos and filenames."
        )

    

def load_images(image_dir: Path, preprocess) -> torch.Tensor:
    check_images_present(image_dir)
    tensors = []
    for name in NAMES:
        with Image.open(image_dir / f"{name}.jpg") as image:
            tensors.append(preprocess(image.convert("RGB")))
    return torch.stack(tensors)


def save_heatmap(scores: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.7))
    image = ax.imshow(scores, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(NAMES)), labels=[x.title() for x in NAMES])
    ax.set_yticks(range(len(NAMES)), labels=[x.title() for x in NAMES])
    for i in range(len(NAMES)):
        for j in range(len(NAMES)):
            ax.text(j, i, f"{scores[i, j]:.3f}", ha="center", va="center", color="black")
    ax.set_title("CLIP image–image cosine similarity")
    fig.colorbar(image, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_experiment(image_dir: Path, output_dir: Path, device: str) -> None:
    check_images_present(image_dir)  # Fail fast before downloading large model weights.
    import open_clip  # Import only when running the actual image experiment.

    print(f"Device: {device}")
    print(f"Loading CLIP {MODEL_NAME} ({PRETRAINED} weights); first run may download weights...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()

    images = load_images(image_dir, preprocess).to(device)
    with torch.inference_mode():
        raw = model.encode_image(images)

    embeddings = normalized_embeddings(raw).cpu()
    similarity = cosine_similarity_matrix(embeddings)
    margin = husky_wolf_margin(similarity).item()

    print(f"Embedding tensor shape: {tuple(embeddings.shape)}")
    print("Order: husky, wolf, retriever")
    print("Similarity matrix:\n", similarity.numpy().round(4))
    print(f"Husky–wolf: {similarity[0, 1].item():.4f}")
    print(f"Husky–retriever: {similarity[0, 2].item():.4f}")
    print(f"Delta (wolf - retriever): {margin:+.4f}")
    print("Reminder: this measures similarities for these images, not the CLIP mechanism.")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "experiment_01_embeddings.npy", embeddings.numpy())
    save_heatmap(similarity.numpy(), output_dir / "experiment_01_similarity.png")
    results = {
        "experiment": "01_image_embedding_similarity",
        "model": MODEL_NAME,
        "pretrained": PRETRAINED,
        "device": device,
        "image_order": list(NAMES),
        "embedding_shape": list(embeddings.shape),
        "similarity_matrix": similarity.tolist(),
        "husky_wolf": similarity[0, 1].item(),
        "husky_retriever": similarity[0, 2].item(),
        "delta_wolf_minus_retriever": margin,
        "caveat": "Single three-image observation. No causal conclusion; no OOD test.",
    }
    (output_dir / "experiment_01_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Results saved to: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=ROOT / "data" / "images")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=choose_device())
    args = parser.parse_args()
    run_experiment(args.images_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()
