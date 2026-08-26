# one-shot-landmark-detector

Conformal prediction sets for one-shot keypoint localization with frozen vision
transformers.

Given a single labeled reference image, a frozen DINOv3 backbone localizes the
corresponding landmark in unseen target images by embedding similarity. This
repository builds **distribution-free conformal prediction sets** on top of those
similarity scores, at three spatial resolutions:

| Level | How embeddings are produced | Forward passes / image | Prediction set granularity |
|---|---|---|---|
| `patch` | One pass; ViT patch tokens used directly | 1 | whole 16×16 patches |
| `bilinear` | One pass; patch grid bilinearly upsampled | 1 | pixels |
| `pixel` | `P²` shifted passes, each pixel is a patch centre in exactly one | 256 | pixels |

> ```bash
> git clone <repo-url> && cd one-shot-landmark-detector
> git checkout avila/experimental
> ```

---

## Installation

Requires **Python ≥ 3.10** and a CUDA-capable GPU (CPU works but pixel-level
scoring will be very slow).

```bash
pip install -e .
```

This installs `torch`, `numpy`, `tqdm`, `transformers`, `Pillow`,
`huggingface_hub`, and `cp4icl` (the conformal-prediction primitives).


### Backbone access

The default backbone is
[`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m),
which is **gated on Hugging Face**. You must accept the model licence on that
page once, then expose a token:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

If `HF_TOKEN` is unset, the code falls back to an *interactive* login prompt,
which will hang non-interactive/batch jobs.

---

## Quickstart

Run one landmark at patch resolution:

```bash
python scripts/run.py \
  -c shared/mini_cephalometric/train_annotations.json \
  -t shared/mini_cephalometric/test_annotations.json \
  --base_img_path /path/to/images \
  -l 1 \
  --level patch \
  --methods caos scos \
  --temperature 0.05 --alpha 0.1 \
  -o results/patch_lm1.csv
```

Run **all** landmarks in a dataset, generating image embeddings only once:

```bash
python scripts/run_all_lms.py \
  --dir shared/mini_cephalometric \
  --base_img_path /path/to/images \
  --level patch \
  --methods caos scos yk_adjust yk_split modsel_cp_ub \
  --temperature 0.05 --alpha 0.1 \
  -o results/patch_all_lms.csv
```

Swap `--level patch` for `bilinear` or `pixel` to compare resolutions.

> Prefer `run_all_lms.py` whenever you need more than one landmark: the ViT
> forward passes dominate runtime and it reuses them across every landmark,
> whereas looping `run.py` recomputes them each time.

---

## Input format

### Annotation files

`train_annotations.json` (calibration) and `test_annotations.json` are JSON
lists, one entry per image:

```json
[
  {"img_path": "train/002.png", "landmarks": [[173, 224], [298, 210], ...]},
  {"img_path": "train/003.png", "landmarks": [[168, 232], [285, 197], ...]}
]
```

- `img_path` — image location **relative to `--base_img_path`**.
- `landmarks` — one `[x, y]` pixel coordinate per landmark, in a **consistent
  order across all images**. `--landmark_idx` / `-l` indexes into this list.

Every image must annotate the same number of landmarks in the same order;
`-l 1` refers to the same anatomical point in every image.

### `metadata.json` (optional)

If present in the `--dir` passed to `run_all_lms.py`, landmark names are read
from it and appear in the output CSV; otherwise landmarks are labelled
`landmark_0`, `landmark_1`, …

```json
{
  "landmark_names": ["sella", "nasion", "orbitale", ...],
  "n_landmarks": 19
}
```

---

## Command-line reference

### Common to both scripts

| Flag | Default | Description |
|---|---|---|
| `--base_img_path` | *required* | Directory prepended to each `img_path` |
| `--level` | `patch` | `patch`, `pixel`, or `bilinear` |
| `--methods` | `caos scos` | Space-separated method names, or `all` |
| `--alpha` | `0.1` | Miscoverage level (`0.1` → 90 % target coverage) |
| `--temperature` | `0.05` | Softmax temperature for nonconformity scores |
| `--k` | `3` | Number of nearest reference predictors averaged by CAOS |
| `--patch_size` | `16` | ViT patch size; must match the backbone |
| `--softmax` / `--no-softmax` | on | Softmax-normalize scores over candidates |
| `--normalize` / `--no-normalize` | on | Mean-centre embeddings before L2 normalization |
| `--device` | `cuda` | Torch device |
| `-o`, `--output_path` | none | Write results as CSV |
| `--verbose` | off | DEBUG-level logging |

### `run.py` only

| Flag | Description |
|---|---|
| `-c`, `--calib_path` | Path to `train_annotations.json` |
| `-t`, `--test_path` | Path to `test_annotations.json` |
| `-l`, `--landmark_idx` | Index of the single landmark to evaluate |

### `run_all_lms.py` only

| Flag | Description |
|---|---|
| `--dir` | Directory holding `train_annotations.json`, `test_annotations.json`, and optionally `metadata.json` |
| `--start_landmark` | Resume from this landmark index; **appends** to `--output_path` instead of overwriting |

Results are flushed after every landmark, so a crash loses at most one
landmark's work. Re-run with `--start_landmark N` to resume.

---

## Conformal methods

| Name | Description |
|---|---|
| `caos` | Aggregates the `k` most confident one-shot predictors |
| `scos` | Split conformal, averaged over individual one-shot predictors |
| `yk_split` | Selects the most efficient predictor using a held-out split |
| `yk_adjust` | Uses all calibration data with a selection-bias adjustment |
| `modsel_cp_ub` | Test-input-aware predictor selection |
| `yk_baseline` | Unadjusted selection baseline |

`caos` and `scos` need only ground-truth-location scores. The selection methods
(`yk_*`, `modsel_cp_ub`) additionally require the **full** score matrix over
every candidate location, which is cheap at `patch` level but is the dominant
cost at `pixel`/`bilinear` level.

> `fullcaos` is registered but **not implemented** — it raises `NameError`.
> Avoid it, and avoid `--methods all`, which includes it.

---

## Output

One CSV row per `(landmark, method)`.

**`run.py`**

```
level, landmark_idx, alpha, temperature, patch_size, apply_softmax, normalize,
k, n_calib, n_test, method, coverage, avg_set_size, total_time_seconds
```

**`run_all_lms.py`**

```
landmark_idx, landmark_name, level, alpha, temperature, patch_size,
apply_softmax, normalize, k, n_calib, n_test, method, coverage, avg_set_size,
landmark_time_seconds
```

- `coverage` — fraction of test images whose true landmark falls inside the
  prediction set. Should be ≥ `1 - alpha`.
- `avg_set_size` — mean prediction-set size. **Units depend on `--level`:**
  **patches** for `patch`, **pixels** for `pixel` and `bilinear`.

### Comparing across resolutions

Because the units differ, divide pixel-level sizes by `patch_size²` (256 for
`P=16`) to obtain **patch-equivalent units (PEU)** before comparing:

```
PEU = avg_set_size / 256      # pixel and bilinear levels
PEU = avg_set_size            # patch level (already in patches)
```

---