#!/usr/bin/env python3
"""Build the README figures in docs/ from samples/ and output/."""
import glob
import os

import cv2
import numpy as np

OUT, DOCS = "output", "docs"
os.makedirs(DOCS, exist_ok=True)
read = lambda p: cv2.imread(p, cv2.IMREAD_GRAYSCALE)
bgr = lambda im: im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
jpg = [cv2.IMWRITE_JPEG_QUALITY, 90]


def label(im, text):
    im = bgr(im).copy()
    cv2.rectangle(im, (0, 0), (14 + 11 * len(text), 30), (0, 0, 0), -1)
    cv2.putText(im, text, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def row(tiles, gap=8):
    pad = np.full((tiles[0].shape[0], gap, 3), 255, np.uint8)
    out = []
    for t in tiles:
        out += [bgr(t), pad]
    return np.hstack(out[:-1])


fused = read(f"{OUT}/fused.png")
single, naive = read(f"{OUT}/single_view.png"), read(f"{OUT}/naive_average.png")

# 1. overview: the four frames with every board copy outlined  ->  the one fused board
T = 440
frames = [cv2.resize(cv2.imread(p), (T, T), interpolation=cv2.INTER_AREA) for p in sorted(glob.glob(f"{OUT}/found_*.png"))]
frames = [label(f, f"frame {i + 1}") for i, f in enumerate(frames)]
g = 8
grid = np.full((2 * T + g, 2 * T + g, 3), 255, np.uint8)
for i, f in enumerate(frames[:4]):
    y, x = (i // 2) * (T + g), (i % 2) * (T + g)
    grid[y:y + T, x:x + T] = f
H = grid.shape[0]
res = cv2.resize(fused, (round(fused.shape[1] * H / fused.shape[0]), H), interpolation=cv2.INTER_AREA)
res = label(res, "result: one whole board")
arrow = np.full((H, 130, 3), 255, np.uint8)
cv2.arrowedLine(arrow, (20, H // 2), (110, H // 2), (0, 120, 255), 8, cv2.LINE_AA, tipLength=0.4)
cv2.imwrite(f"{DOCS}/overview.jpg", np.hstack([grid, arrow, res]), jpg)

# 2. the result
cv2.imwrite(f"{DOCS}/fused.png", fused)

# 3. one view / plain average / final, at full output resolution
crops = {"detail_qfn": (800, 840), "detail_connector": (810, 1630), "detail_ics": (1220, 400)}
for name, (x, y) in crops.items():
    s = 420
    tiles = [label(im[y:y + s, x:x + s], t) for im, t in
             ((single, "single shot"), (naive, "plain average"), (fused, "xray-stack"))]
    cv2.imwrite(f"{DOCS}/{name}.png", row(tiles))

# 4. how many copies cover each pixel, and how many were actually used
cov, eff = read(f"{OUT}/coverage.png"), read(f"{OUT}/effective_views.png")
small = lambda im: cv2.resize(im, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
cv2.imwrite(f"{DOCS}/views.jpg", row([label(small(cov), "copies that see each spot"),
                                      label(small(eff), "copies actually used")]), jpg)

# 4b. registration: one far-away copy's difference from the final image, before and after flow correction
if os.path.exists(f"{OUT}/parallax_before.png"):
    before, after = cv2.imread(f"{OUT}/parallax_before.png"), cv2.imread(f"{OUT}/parallax_after.png")
    ys, xs = np.nonzero(before.max(2) > 12)
    x0, x1, y0, y1 = np.percentile(xs, 1).astype(int), np.percentile(xs, 99).astype(int), np.percentile(ys, 1).astype(int), np.percentile(ys, 99).astype(int)
    f = 900 / (y1 - y0)
    shrink = lambda im: cv2.resize(im[y0:y1, x0:x1], None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{DOCS}/registration.png", row([label(shrink(before), "affine placement only"),
                                                 label(shrink(after), "after flow correction")]))

# 5. the detector's brightness falloff, as estimated (one panel per frame)
if os.path.exists(f"{OUT}/flatfield.png"):
    ff = read(f"{OUT}/flatfield.png")
    cv2.imwrite(f"{DOCS}/flatfield.jpg", cv2.resize(ff, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA), jpg)


# 6. the noise numbers quoted in the README: robust std of the high-pass residual on bare board
def noise(im):
    im = im.astype(np.float32)
    hp = im - cv2.GaussianBlur(im, (0, 0), 4)
    flat = cv2.GaussianBlur(np.abs(cv2.Laplacian(cv2.GaussianBlur(fused.astype(np.float32), (0, 0), 6), cv2.CV_32F)), (0, 0), 6)
    sel = (fused > 120) & (fused < 215) & (flat < np.percentile(flat, 30)) & (single < 250)
    return 1.4826 * np.median(np.abs(hp[sel] - np.median(hp[sel])))


for n, im in (("single shot", single), ("plain average", naive), ("xray-stack", fused)):
    print(f"{n:14s} noise sigma = {noise(im):.2f} grey levels")
