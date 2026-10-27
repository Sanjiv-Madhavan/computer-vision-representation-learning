# Probe 1 — Visual embeddings and representation learning

**Scientific question for Experiment 01:** For these particular images, is a husky's CLIP image embedding closer (by cosine similarity) to a visually similar wolf or to a golden retriever?

This first experiment is an **instrument check**, not a result about how CLIP works in general. We will learn to load a frozen pretrained image encoder, inspect embedding dimensions, calculate similarity, and record observations carefully. More images and controlled comparisons come later.

## Prerequisites

- PyCharm (any edition with Python support)
- Python 3.11 or 3.12 (recommended)
- Internet access on first model run to download the CLIP weights
- No GPU required; the script runs on CPU (slower) or CUDA if available

## Setup in PyCharm (Windows-friendly)

1. Extract the ZIP, then choose **File > Open** in PyCharm and select the extracted `probe1_visual_embeddings` directory (the one containing this README).
2. Configure a Python virtual environment: **Settings > Project > Python Interpreter**. Select or create a project `.venv` using Python 3.11/3.12.
3. Open PyCharm **Terminal**. The terminal should point to the project root.
4. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

   For NVIDIA GPU acceleration, check the [official PyTorch installer](https://pytorch.org/get-started/locally/) for the right CUDA build. The default installation is sufficient to start on CPU.
5. Test the similarity mathematics without downloading model weights:

   ```powershell
   python -m pytest -q
   ```

6. Put three **photographs you have permission to use** in `data/images/` with these exact filenames:

   - `husky.jpg`
   - `wolf.jpg`
   - `retriever.jpg`

   For a fairer preliminary comparison, try similar framing/subject size and note important differences in backgrounds and poses.
7. Run:

   ```powershell
   python -m probe1.experiment_01
   ```

8. Inspect the printed scores and the outputs:

   - `outputs/experiment_01_results.json` — measured values and device/model details
   - `outputs/experiment_01_similarity.png` — three-by-three heatmap
   - `outputs/experiment_01_embeddings.npy` — three normalized 512-D embeddings

The first run will download pretrained OpenAI CLIP ViT-B/32 weights via OpenCLIP. It can take longer on CPU. No fine-tuning takes place.

## Prediction to test

> **User hypothesis (before observing the scores):** A visually similar grey wolf will have higher cosine similarity to a husky image than a golden retriever does, because CLIP's pretrained visual features may respond strongly to their resemblance.

Let `S_HW = sim(husky, wolf)` and `S_HD = sim(husky, retriever)`; our comparison is `delta = S_HW - S_HD`.

- `delta > 0`: wolf is closer for these specific photographs.
- `delta < 0`: retriever is closer for these specific photographs.
- `delta ≈ 0`: similarities are very close for these particular inputs.

A single three-image comparison **cannot identify the model's mechanism** or prove whether 'visual' or 'semantic' information dominates. CLIP features may capture both. Backgrounds, image selection, texture and framing are confounds. The script does not perform classification, calibrate confidence or measure out-of-distribution detection.

## What do we study next?

1. Extend to multiple images per animal, with metadata on source/background/pose; compare distributions rather than single scores.
2. Introduce text embeddings and zero-shot label scores.
3. Hold out animal classes and evaluate potential OOD signals and baselines.
4. Propose controlled experiments to distinguish competing explanations.

Write observations in `RESEARCH_LOG.md` as **measured result**, **interpretation**, **alternative explanation**, and **next test**.

## Working with Codex

This project includes `AGENTS.md` (working principles) and `CODEX_HANDOFF.md` (a ready-to-copy starting prompt). In PyCharm, open **AI Chat > Codex**, open this folder, and paste the contents of `CODEX_HANDOFF.md` into your first Codex message. Codex can then modify and run the files in this project. Review its proposed edits and command execution.
