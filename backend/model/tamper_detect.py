"""
Image Tampering Detection - core forensic engine.

Approach
--------
Real (untouched) camera JPEGs compress fairly uniformly across the whole
frame. When a region of an image has been copy-pasted, spliced, cloned,
or re-touched, that region has usually been through a *second* round of
JPEG compression (or was never compressed the same way as the rest of
the photo). Error Level Analysis (ELA) exploits this: resave the image
at a known JPEG quality, diff it against the original, and regions that
were edited "light up" because they lose/gain error at a different rate
than the rest of the image.

We combine ELA with two supporting signals so a single accidental
bright edge doesn't cause a false positive:
  1. ELA hot-region ratio       - how much of the image has abnormally
                                   high recompression error
  2. Noise-inconsistency score  - local noise variance should be fairly
                                   uniform in a real photo; spliced
                                   regions usually have different noise
  3. Edge-artifact score        - unnatural hard edges around a pasted
                                   region show up as high-frequency
                                   discontinuities in the ELA map

The three are combined into a single confidence score. This is a
classical (non-deep-learning) forensics pipeline, so it needs no
training data and is fully explainable - useful for a demo API.
"""

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops
import cv2


ELA_QUALITY = 90          # JPEG quality used for recompression
ELA_SCALE = 15            # brightness scale so diffs are visible
TAMPER_THRESHOLD = 0.34   # combined-score cutoff for "Tampered"


@dataclass
class TamperResult:
    label: str
    confidence: float
    ela_score: float
    noise_score: float
    edge_score: float
    hot_region_pct: float
    ela_image_b64: str


def _load_image(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img


def compute_ela(img: Image.Image, quality: int = ELA_QUALITY) -> Image.Image:
    """Resave the image at a fixed JPEG quality and diff against original."""
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    diff = ImageChops.difference(img, resaved)
    extrema = diff.getextrema()
    max_diff = max(ex[1] for ex in extrema) or 1
    scale = 255.0 / max_diff

    ela_img = Image.eval(diff, lambda px: min(255, int(px * scale)))
    return ela_img


def _hot_region_ratio(ela_np: np.ndarray) -> float:
    """Fraction of pixels whose ELA error is far above the image's own median."""
    gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    med = np.median(gray)
    mad = np.median(np.abs(gray - med)) + 1e-6
    hot_mask = gray > (med + 6 * mad)
    return float(hot_mask.mean())


def _noise_inconsistency(img_np: np.ndarray) -> float:
    """Std-dev of local noise variance across 16x16 blocks, normalized."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    h, w = lap.shape
    block = 16
    variances = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            patch = lap[y:y + block, x:x + block]
            variances.append(patch.var())
    variances = np.array(variances) if variances else np.array([0.0])
    if variances.mean() < 1e-6:
        return 0.0
    cv_ratio = variances.std() / (variances.mean() + 1e-6)
    return float(min(cv_ratio / 4.0, 1.0))  # normalize to ~0-1


def _edge_artifact_score(ela_np: np.ndarray) -> float:
    """High-frequency discontinuities in the ELA map itself (pasted-region edges)."""
    gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.mean() / 255.0)


def _pil_to_b64(img: Image.Image) -> str:
    import base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def analyze_image(image_bytes: bytes) -> TamperResult:
    img = _load_image(image_bytes)

    # Downscale very large images for speed; forensic signal is preserved.
    max_dim = 1024
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))

    ela_img = compute_ela(img)
    ela_np = np.array(ela_img)
    img_np = np.array(img)

    ela_score = _hot_region_ratio(ela_np)
    noise_score = _noise_inconsistency(img_np)
    edge_score = _edge_artifact_score(ela_np)

    # Weighted combination. Noise-inconsistency turned out to be the most
    # reliable discriminator in calibration testing, followed by the ELA
    # hot-region ratio; the edge-artifact score is the noisiest of the
    # three (gradients/textures can trigger it) so it gets the least weight.
    combined = 0.40 * ela_score + 0.45 * noise_score + 0.15 * edge_score
    confidence = float(min(combined / TAMPER_THRESHOLD, 1.0)) if combined <= TAMPER_THRESHOLD \
        else float(min(0.5 + (combined - TAMPER_THRESHOLD), 1.0))

    label = "Tampered" if combined >= TAMPER_THRESHOLD else "Authentic"
    # confidence: how far the combined score is from the decision boundary
    confidence = float(min(0.55 + abs(combined - TAMPER_THRESHOLD) * 1.8, 0.99))

    return TamperResult(
        label=label,
        confidence=round(confidence, 3),
        ela_score=round(ela_score, 4),
        noise_score=round(noise_score, 4),
        edge_score=round(edge_score, 4),
        hot_region_pct=round(ela_score * 100, 2),
        ela_image_b64=_pil_to_b64(ela_img),
    )
