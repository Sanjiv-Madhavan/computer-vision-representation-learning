# Concepts for Probe 1: visual embeddings

This is a learning guide for our first three-image experiment. Read it in order,
alongside [the extraction script](../probe1/experiment_01.py) and
[the similarity functions](../probe1/similarity.py). The shapes below describe
the expected pipeline, not measured experimental outputs.

Our question is: for these particular photographs, is a husky's CLIP image
embedding closer to a wolf's or a golden retriever's?

## 1. Images as numerical arrays

An RGB image represents each pixel with red, green, and blue values. A tensor is
a numerical array that can have multiple dimensions.

PyTorch's image convention in this scaffold is:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| One preprocessed image | `[3, 224, 224]` | Channels, height, width |
| Three stacked images | `[3, 3, 224, 224]` | Images, channels, height, width |
| Three CLIP embeddings | `[3, 512]` | Images, learned features |
| Similarity matrix | `[3, 3]` | Every image compared with every image |

`torch.stack(tensors)` adds the image dimension. The image order is always
`[husky, wolf, retriever]`. The first two dimensions of the image batch both
happen to be three, but they have different meanings.

**What I need to understand:** Explain what every dimension represents and
identify which row belongs to each photograph.

## 2. Image preprocessing

The pretrained encoder expects a particular input format. The script converts
each image to RGB and uses the model's evaluation preprocessing: resize, center
crop to `224 × 224`, convert pixels to a tensor, and standardize each channel.
Resizing changes the sampling of the pixels; cropping selects a region and can
remove part of the subject or background.

For an ordinary 8-bit RGB pixel value `p`, channel standardization is:

```text
x_c = (p_c / 255 - mean_c) / std_c
```

The mean and standard deviation are fixed constants supplied with the model's
preprocessing configuration. We do not estimate them from our three photographs.
Standardized pixel values can be negative or greater than one.

This pixel preprocessing is a different operation from the embedding L2
normalization in section 7.

**What I need to understand:** Explain why inputs must match the pretrained
model's expected format and why the crop can affect our comparison.

## 3. Features, encoders, and embeddings

A feature is an aspect of the input represented numerically. An encoder is a
function that transforms an input into a representation. An embedding is the
resulting vector of coordinates.

```text
raw_embedding = image_encoder(preprocessed_image)
```

For this CLIP model, each output vector has 512 coordinates. These are learned
coordinates; we cannot assume that individual coordinates mean "ears", "fur",
or another named property. The embedding describes the whole processed image,
so background and framing may affect it too.

**What I need to understand:** Distinguish an image's pixels from its embedding,
and explain why `[3, 512]` means three vectors rather than 512 class predictions.

## 4. Pretraining and CLIP

Pretraining learns model parameters before our experiment. CLIP uses image and
text encoders trained with a contrastive objective: representations of matching
image–text pairs are encouraged to align relative to mismatched pairs.

We use the original OpenAI pretrained weights through OpenCLIP, with model name
`ViT-B-32` and pretrained setting `openai`. In this experiment we only call the
image encoder; no text prompts or class labels are encoded.

CLIP's training provides context for interpreting its representations. It does
not establish that our image comparison isolates semantic or visual information.

**What I need to understand:** Distinguish pretraining from our inference run,
and describe the contrastive learning idea without needing its loss equation yet.

## 5. Vision Transformers and patches

ViT means Vision Transformer. ViT-B/32 uses `32 × 32` pixel patches. A
`224 × 224` input contains a `7 × 7` grid, or 49 image patches. The model maps
patches into internal representations and uses attention to combine information
across positions before producing the image embedding.

The 32 in the model name is the patch width and height, not the final embedding
dimension. The final image embedding has 512 coordinates.

**What I need to understand:** Explain how patches connect the image to a
sequence of representations. Detailed attention equations can wait.

## 6. Frozen inference

Frozen inference means applying learned parameters without updating them.

In `run_experiment`:

- `model.eval()` selects evaluation behavior for applicable layers. It does not
  itself disable gradients or change parameter gradient flags.
- `torch.inference_mode()` disables gradient tracking during encoding.
- `model.encode_image(images)` computes the embeddings.
- There is no optimizer, backward pass, or parameter update.

`.to(device)` moves tensors to the selected processor. `.cpu()` moves results
to CPU memory for saving. These are placement operations, not new similarity
calculations. `.float()` converts embeddings to float32 before normalization.

**What I need to understand:** Explain why running a pretrained model does not
necessarily train it, and distinguish evaluation mode from gradient tracking.

## 7. Vector geometry and similarity

### L2 length and normalization

For a nonzero embedding `e`, its L2 length is the square root of the sum of its
squared coordinates. Normalization divides each coordinate by that length:

```text
length(e) = sqrt(sum_k e_k²)
unit_e = e / length(e)
```

For a mathematical example, `[3, 4]` has length 5 and normalizes to `[0.6, 0.8]`.
This preserves direction while removing magnitude. It is not an image result.

`F.normalize(embeddings.float(), p=2, dim=-1)` performs this operation across the
features of each row. PyTorch uses an epsilon safeguard for very small norms;
a zero vector has no defined cosine direction and does not become unit length.

**What I need to understand:** Normalize a small nonzero vector by hand and
explain why `dim=-1` operates across the 512 features rather than across images.

### Dot products, cosine similarity, and matrix multiplication

A dot product multiplies corresponding coordinates and sums the products:

```text
dot(a, b) = sum_k a_k * b_k
cosine(a, b) = dot(a, b) / (length(a) * length(b))
```

After normalizing nonzero vectors, both lengths are one, so their dot product
equals cosine similarity. If `E` contains our normalized rows:

```text
S = E @ E.T
[3, 512] @ [512, 3] -> [3, 3]
S[i, j] = sum_k E[i, k] * E[j, k]
```

`.T` transposes the matrix; `@` performs matrix multiplication. The helper
`cosine_similarity_matrix` normalizes its input itself, so the script normalizes
the already normalized rows again before multiplication.

For nonzero embeddings, cosine values lie between -1 and 1 mathematically.
Floating-point arithmetic can introduce small numerical discrepancies. The
matrix should be symmetric and its diagonal approximately one. Higher values
mean more aligned directions; they are not class probabilities or calibrated
confidence scores.

### The comparison margin

```text
delta = S[0, 1] - S[0, 2]
      = similarity(husky, wolf) - similarity(husky, retriever)
```

- Positive: wolf is closer for these photographs.
- Negative: retriever is closer for these photographs.
- Near zero: the two similarities are close; we have not defined a significance
  threshold or estimated uncertainty from these three inputs.

`.item()` converts the one-element margin tensor into a Python number. Saving
the embeddings as `.npy`, the measurements as `.json`, and the heatmap as `.png`
records the result; it does not introduce another similarity measure.

**What I need to understand:** Explain each matrix dimension, identify the two
entries being subtracted, and interpret the sign without treating it as confidence.

## 8. Experimental reasoning and confounds

Keep these categories separate in our notes:

| Category | Meaning in this probe |
| --- | --- |
| Established model facts | Architecture, pretrained weights, and preprocessing configuration |
| Our hypothesis | Before viewing scores, we predict higher husky–wolf similarity |
| Measured observations | Values actually produced from our chosen images |
| Authors' interpretations | Claims from a source, attributed to its authors when discussed |
| Our interpretations | Explanations we propose after observing results |
| Open questions | Explanations or uncertainties this comparison cannot resolve |

Your pre-registered hypothesis remains in [README](../README.md) and
[RESEARCH_LOG](../RESEARCH_LOG.md): the wolf will be more similar to the husky,
motivated by visually shared traits and the learned representation. That is a
prediction, not an established mechanism.

Background, subject size, pose, lighting, texture, image source, and the center
crop can all affect the comparison. One photograph per animal cannot establish
general behavior across animals or separate visual from semantic information.
We also do not measure classification accuracy, calibrated confidence, or
out-of-distribution detection here.

**What I need to understand:** State the result as an observation about these
inputs, distinguish it from an explanation, and identify plausible confounds.

## How tests support the experiment

Follow test-driven development for functional changes: write a meaningful test,
confirm the expected failure, make the smallest passing change, and refactor
with relevant tests passing.

Our [four baseline tests](../tests/test_similarity.py) check normalization,
similarity-matrix properties, margin calculation, and invalid embedding shape.
They use synthetic tensors and require no photographs or model weights. They
do not yet test the pretrained encoder's forward pass or the full image pipeline.

Correctness tests should enforce mathematical and pipeline contracts. They
must not require `delta > 0`: an observation that contradicts our hypothesis
can still come from correct code. Checks requiring model weights and user
photographs should remain separate from offline unit tests.

## Foundations needed now

You need vectors, sums, square roots, division, matrix shapes, and subtraction.
Calculus, backpropagation, detailed attention derivations, and training a model
are not prerequisites for this extraction. Learn the concepts above alongside
the existing code; the current scope remains exactly three images, frozen
encoding, cosine similarities, the margin, and saved outputs.
