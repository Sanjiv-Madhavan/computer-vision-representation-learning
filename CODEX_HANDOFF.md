# First message to Codex (paste into PyCharm AI Chat > Codex)

We are beginning **Probe 1: Visual Embeddings & Representation Learning**, a hands-on exploratory research project after surveying six CVPR 2026 papers. I want to learn how pretrained CLIP visual representations behave before studying zero-shot classification, confidence, unfamiliar classes and OOD detection.

I have opened this project in PyCharm. Please:

1. Read README.md, AGENTS.md, RESEARCH_LOG.md, and the code.
2. Review the PyTorch scaffold for errors, clarity, and reproducibility, making only targeted fixes if needed.
3. Help me configure my PyCharm interpreter and run `python -m pytest -q`.
4. Explain the shape of the embedding tensor, why L2 normalization matters, and why `E @ E.T` computes cosine similarity.
5. Tell me exactly how to place `husky.jpg`, `wolf.jpg`, and `retriever.jpg` in data/images/ and run `python -m probe1.experiment_01`.
6. Ask for my predictions before interpreting actual scores. Do not fabricate outputs or silently run additional scientific experiments.

My pre-registered prediction: a husky's image embedding will be closer to a visually similar grey wolf than to a golden retriever. This is a hypothesis, not an established mechanism. I want to look for confounds and alternative explanations after seeing the data.

Keep the pace incremental and use PyTorch rather than hiding everything behind a high-level API.
