# Research log: Probe 1

## Experiment 01 — Three-image embedding sanity check

**Question:** For these chosen photographs, is a husky embedding more cosine-similar to a wolf or a golden retriever?

**Pre-registered hypothesis:** The wolf will be more similar to the husky, because of visually shared traits and the representation learned in CLIP pretraining.

**Model:** CLIP ViT-B/32 with original OpenAI pretrained weights, via `open_clip_torch`. Frozen encoder; no training.

**Inputs:** (Fill in image provenance, photo conditions, framing, and date of collection.)
- husky.jpg:
- wolf.jpg:
- retriever.jpg:

**Measured results:** Not run yet.
- Husky–wolf cosine similarity:
- Husky–retriever cosine similarity:
- Delta (wolf minus retriever):

**Initial interpretation:** To be written *after* measurements.

**Alternative explanations / confounds:** To investigate after observing outputs (e.g., backgrounds, object size, pose, style, textures, and semantic signals).

**What the result cannot establish:** A causal or general separation between visual and semantic representations; performance on OOD detection; calibrated confidence; behavior over the population of animal images.

**Next discriminating experiment:** To be decided after examining data.
