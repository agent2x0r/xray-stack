#!/usr/bin/env python3
"""Fuse many X-ray views of the *same* PCB design into one clean image.

A panel of identical boards photographed a few times gives a dozen-plus
independent looks at one design.  This script finds every copy, registers it
onto a common canvas, corrects the residual cone-beam parallax with dense
optical flow, and averages only the copies that agree with a reference view.

See README.md for the full write-up.
"""
import argparse
import os

import cv2
import numpy as np


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def frame_mask(shape, margins):
    """Valid-pixel mask for a source image: 0 on the detector's frame lines."""
    left, top, right, bottom = margins
    m = np.ones(shape, np.float32)
    m[:, :left] = 0
    m[:top] = 0
    if right:
        m[:, -right:] = 0
    if bottom:
        m[-bottom:] = 0
    return m


def to_u8(a, mask):
    """Float canvas -> uint8, with everything outside `mask` painted white."""
    return np.where(mask, np.clip(np.nan_to_num(a, nan=255.0), 0, 255), 255).astype(np.uint8)


def match_exposure(a, mask, ref, ref_mask):
    """Linear gain/offset so `a` matches `ref` (ignoring clipped pixels)."""
    ok = mask & ref_mask & (a < 245) & (ref < 245) & (a > 5) & (ref > 5)
    if ok.sum() < 1000:
        return a
    g, b = np.polyfit(a[ok].astype(np.float64), ref[ok].astype(np.float64), 1)
    return (a * g + b).astype(np.float32)


def clipped_mean(stack, weights=None, k=2.5, anchor=None):
    """Per-pixel (weighted) mean of a NaN-padded stack after MAD outlier rejection.

    `anchor` names one layer that is trusted when nothing else is: wherever the
    other layers carry almost no weight, the anchor is kept even if it looks
    like an outlier against the (then meaningless) median.
    """
    med = np.nanmedian(stack, axis=0)
    mad = np.nanmedian(np.abs(stack - med), axis=0) * 1.4826 + 2.0
    keep = np.abs(stack - med) < k * mad            # NaN compares False
    w = keep.astype(np.float32) if weights is None else np.where(keep, weights, 0).astype(np.float32)
    if anchor is not None:
        lonely = (w.sum(0) - w[anchor] < 1.0) & np.isfinite(stack[anchor])
        w[anchor][lonely] = 1.0 if weights is None else weights[anchor][lonely]
    wsum = w.sum(0)
    out = (np.nan_to_num(stack) * w).sum(0) / np.maximum(wsum, 1e-6)
    out[wsum <= 1e-6] = 255
    n_eff = wsum ** 2 / np.maximum((w ** 2).sum(0), 1e-6)   # effective number of views
    return np.clip(out, 0, 255).astype(np.float32), n_eff


# --------------------------------------------------------------------------
# stage 1: find every copy of the board (feature matching + sequential RANSAC)
# --------------------------------------------------------------------------
def find_instances(sift, ref, ref_mask, feats, full_affine, min_inliers):
    """Locate all copies of `ref` in every image.

    Returns a list of (image_index, 2x3 affine image->canvas, inliers, median
    residual in px).  Matching runs image -> reference: the reference holds a
    single copy of the board, so Lowe's ratio test stays meaningful even
    though each image contains several identical copies.
    """
    kr, dr = sift.detectAndCompute(ref, ref_mask)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    found = []
    for idx, (ki, di) in enumerate(feats):
        good = [m for m, n in bf.knnMatch(di, dr, k=2) if m.distance < 0.75 * n.distance]
        src = np.float32([ki[m.queryIdx].pt for m in good])
        dst = np.float32([kr[m.trainIdx].pt for m in good])
        # Sequential RANSAC: fit one copy, remove its inliers, fit the next.
        while len(src) >= min_inliers:
            A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                 ransacReprojThreshold=2.0, maxIters=5000)
            if A is None:
                break
            inl = inl.ravel().astype(bool)
            if inl.sum() < min_inliers:
                break
            if full_affine:      # polish: full 6-DoF affine on the consensus set
                A2, inl2 = cv2.estimateAffine2D(src[inl], dst[inl], method=cv2.RANSAC,
                                                ransacReprojThreshold=1.0)
                if A2 is not None:
                    k = inl2.ravel().astype(bool)
                    A3, _ = cv2.estimateAffine2D(src[inl][k], dst[inl][k], method=cv2.LMEDS)
                    A = A3 if A3 is not None else A2
            scale = np.sqrt(abs(np.linalg.det(A[:, :2])))
            duplicate = any(i == idx and np.hypot(*(B[:, 2] - A[:, 2])) < 40 for i, B, _, _ in found)
            if 0.98 < scale < 1.02 and not duplicate:   # reject degenerate / repeated fits
                res = np.linalg.norm(src[inl] @ A[:, :2].T + A[:, 2] - dst[inl], axis=1)
                found.append((idx, A, int(inl.sum()), float(np.median(res))))
            src, dst = src[~inl], dst[~inl]
    return found


# --------------------------------------------------------------------------
# deskew: measure how far the board is rotated on the canvas
# --------------------------------------------------------------------------
def estimate_tilt(im, min_len=30):
    """Board tilt in degrees, from the straight edges in the image.

    PCB artwork is drawn almost entirely at 0, 45 and 90 degrees.  Detect line
    segments (LSD, sub-pixel), fold their angles modulo 45 degrees so all
    three families vote for the same number, and take a length-weighted mean
    around the length-weighted median.  Negative = rotated counter-clockwise
    as displayed.  Returns (tilt, number of segments used).
    """
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    seg = np.asarray(lsd.detect(cv2.GaussianBlur(im, (0, 0), 1.0))[0]).reshape(-1, 4)
    d = seg[:, 2:] - seg[:, :2]
    length = np.hypot(d[:, 0], d[:, 1])
    ang = (np.degrees(np.arctan2(d[:, 1], d[:, 0])) + 22.5) % 45 - 22.5
    keep = length > min_len
    if keep.sum() < 10:
        return 0.0, int(keep.sum())
    ang, length = ang[keep], length[keep]
    order = np.argsort(ang)
    cum = np.cumsum(length[order])
    median = ang[order][np.searchsorted(cum, cum[-1] / 2)]
    near = np.abs(ang - median) < 1.0
    return float((ang[near] * length[near]).sum() / length[near].sum()), int(near.sum())


# --------------------------------------------------------------------------
# flat-field self-calibration
# --------------------------------------------------------------------------
def estimate_flatfield(imgs, margins, inst, canvas_wh, degree=3, step=6):
    """Estimate the detector's smooth brightness falloff from the data itself.

    The same board feature is seen at many detector positions.  In the log
    domain   log I_k(x) = b(x) + f(p_k(x)) + g_image   where b is the board,
    f a low-order polynomial over detector coordinates p and g a per-image
    exposure.  Subtracting the per-feature mean over copies removes b, which
    leaves a linear least-squares problem in the polynomial coefficients.
    Returns one multiplicative field per image (divide the image by it).
    """
    CW, CH = canvas_wh
    h, w = imgs[0].shape
    terms = [(a, b) for a in range(degree + 1) for b in range(degree + 1) if 1 <= a + b <= degree]
    xs, ys = np.meshgrid(np.arange(0, CW, step, dtype=np.float32), np.arange(0, CH, step, dtype=np.float32))
    L, V, U, Vv, J = [], [], [], [], []
    for idx, A, _, _ in inst:
        im = imgs[idx].astype(np.float32)
        ok = frame_mask(im.shape, margins) * (cv2.dilate((im >= 245).astype(np.uint8), np.ones((15, 15), np.uint8)) == 0)
        ok *= cv2.erode((im > 12).astype(np.uint8), np.ones((9, 9), np.uint8))
        bw = cv2.GaussianBlur(ok, (0, 0), 3)
        bl = cv2.GaussianBlur(im * ok, (0, 0), 3) / np.maximum(bw, 1e-3)     # masked local mean
        Ainv = cv2.invertAffineTransform(A)
        px = (Ainv[0, 0] * xs + Ainv[0, 1] * ys + Ainv[0, 2]).astype(np.float32)
        py = (Ainv[1, 0] * xs + Ainv[1, 1] * ys + Ainv[1, 2]).astype(np.float32)
        val = cv2.remap(bl, px, py, cv2.INTER_LINEAR, borderValue=0)
        good = (cv2.remap(bw, px, py, cv2.INTER_LINEAR, borderValue=0) > 0.9) & (val > 12)
        L.append(np.log(np.maximum(val, 1)).ravel()); V.append(good.ravel())
        U.append((2 * px / w - 1).ravel()); Vv.append((2 * py / h - 1).ravel()); J.append(idx)
    L, V, U, Vv = map(np.stack, (L, V, U, Vv))
    V &= V.sum(0) >= 2
    n = np.maximum(V.sum(0), 1)
    demean = lambda Q: Q - (np.where(V, Q, 0).sum(0) / n)
    cols = [demean(U ** a * Vv ** b) for a, b in terms]
    cols += [demean(np.repeat((np.array(J) == j).astype(np.float64)[:, None], L.shape[1], 1))
             for j in range(1, len(imgs))]
    X = np.stack([c[V] for c in cols], 1)
    y = demean(L)[V]
    keep = np.ones(len(y), bool)
    for _ in range(4):                                   # robust: drop residual outliers, refit
        coef, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        r = y - X @ coef
        keep = np.abs(r) < 2.5 * 1.4826 * np.median(np.abs(r[keep])) + 1e-6
    rms0, rms1 = np.sqrt(np.mean(y[keep] ** 2)), np.sqrt(np.mean(r[keep] ** 2))
    print(f"flat-field: copy-to-copy disagreement {100 * rms0:.2f}% -> {100 * rms1:.2f}% rms "
          f"({keep.sum()} samples, degree {degree})")
    uu, vv = np.meshgrid(np.linspace(-1, 1, w, dtype=np.float32), np.linspace(-1, 1, h, dtype=np.float32))
    poly = sum(c * uu ** a * vv ** b for c, (a, b) in zip(coef, terms))
    gains = np.concatenate([[0.0], coef[len(terms):]])
    fields = [np.exp(poly + g) for g in gains]
    norm = np.mean([f.mean() for f in fields])
    return [f / norm for f in fields]


# --------------------------------------------------------------------------
# stage 2: warping (affine, optionally composed with a dense flow field)
# --------------------------------------------------------------------------
class Warper:
    def __init__(self, imgs, margins, canvas_wh, scale):
        self.imgs = [im.astype(np.float32) for im in imgs]
        self.masks = [frame_mask(im.shape, margins) for im in imgs]
        self.W, self.H = canvas_wh[0] * scale, canvas_wh[1] * scale
        self.S = np.diag([scale, scale, 1.0])
        self.X, self.Y = np.meshgrid(np.arange(self.W, dtype=np.float32),
                                     np.arange(self.H, dtype=np.float32))

    def __call__(self, idx, A, flow=None):
        """Resample image `idx` onto the canvas in a single interpolation.

        `flow` (canvas px) is added to the canvas coordinates *before* they
        are pushed through the inverse affine, so affine + parallax correction
        cost one Lanczos resample instead of two.
        """
        Minv = np.linalg.inv(self.S @ np.vstack([A, [0, 0, 1]]))
        x, y = (self.X, self.Y) if flow is None else (self.X + flow[..., 0], self.Y + flow[..., 1])
        mx = (Minv[0, 0] * x + Minv[0, 1] * y + Minv[0, 2]).astype(np.float32)
        my = (Minv[1, 0] * x + Minv[1, 1] * y + Minv[1, 2]).astype(np.float32)
        out = cv2.remap(self.imgs[idx], mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        m = cv2.remap(self.masks[idx], mx, my, cv2.INTER_LINEAR, borderValue=0) > 0.999
        m = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        return out, m


def smooth_flow(flow, valid):
    """Normalised-convolution smoothing; extrapolates flow beyond `valid`.

    A small kernel regularises the flow where it was measured; two much wider
    kernels (run on an 8x-reduced copy, they only carry low frequencies)
    extend it smoothly into regions the reference did not cover.
    """
    w = valid.astype(np.float32)
    h, wd = w.shape
    out = np.zeros_like(flow)
    filled = np.zeros(w.shape, bool)
    small_w = cv2.resize(w, (wd // 8, h // 8), interpolation=cv2.INTER_AREA)
    small_f = cv2.resize(flow * w[..., None], (wd // 8, h // 8), interpolation=cv2.INTER_AREA)
    for sigma, thresh in ((5, 0.3), (40, 0.02), (150, 1e-4)):
        if sigma <= 5:
            bw = cv2.GaussianBlur(w, (0, 0), sigma)
            bf = cv2.GaussianBlur(flow * w[..., None], (0, 0), sigma)
        else:
            bw = cv2.resize(cv2.GaussianBlur(small_w, (0, 0), sigma / 8), (wd, h), interpolation=cv2.INTER_LINEAR)
            bf = cv2.resize(cv2.GaussianBlur(small_f, (0, 0), sigma / 8), (wd, h), interpolation=cv2.INTER_LINEAR)
        use = (bw > thresh) & ~filled
        out[use] = (bf / np.maximum(bw, 1e-6)[..., None])[use]
        filled |= use
    return out


# --------------------------------------------------------------------------
# main pipeline
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="X-ray images of the panel (any OpenCV-readable format)")
    ap.add_argument("-o", "--out", default="output", help="output directory")
    ap.add_argument("--seed-image", type=int, default=2, help="index (0-based) of the image holding the seed board")
    ap.add_argument("--seed-box", type=int, nargs=4, default=[690, 360, 1537, 1400],
                    metavar=("X0", "Y0", "X1", "Y1"), help="rough box around one board in the seed image")
    ap.add_argument("--seed-offset", type=int, nargs=2, default=[60, 60], metavar=("X", "Y"),
                    help="where the seed box lands on the canvas")
    ap.add_argument("--canvas", type=int, nargs=2, default=[1150, 1200], metavar=("W", "H"),
                    help="canvas size in source pixels (one board + margin)")
    ap.add_argument("--crop", type=int, nargs=4, default=[60, 40, 1085, 1100],
                    metavar=("X0", "Y0", "X1", "Y1"), help="final crop on the canvas, source pixels")
    ap.add_argument("--scale", type=int, default=2, help="output upsampling factor")
    ap.add_argument("--margins", type=int, nargs=4, default=[6, 4, 4, 12],
                    metavar=("L", "T", "R", "B"), help="source-image border to ignore (detector frame lines)")
    ap.add_argument("--min-inliers", type=int, default=20)
    ap.add_argument("--deskew", default="auto", metavar="auto|off|DEG",
                    help="straighten the board: measure its tilt (auto), leave it (off), or give the tilt in degrees")
    ap.add_argument("--no-flatfield", action="store_true", help="skip the flat-field self-calibration")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in args.images]
    for p, im in zip(args.images, imgs):
        if im is None:
            raise SystemExit(f"cannot read {p}")
    CW, CH = args.canvas
    sc = args.scale

    sift = cv2.SIFT_create(contrastThreshold=0.02)
    feats = [sift.detectAndCompute(im, None) for im in imgs]

    # ---- seed: paste one board onto the canvas as the first reference -----
    x0, y0, x1, y1 = args.seed_box
    x1, y1 = min(x1, imgs[args.seed_image].shape[1]), min(y1, imgs[args.seed_image].shape[0])
    ox, oy = args.seed_offset
    ref = np.full((CH, CW), 255, np.uint8)
    ref_mask = np.zeros((CH, CW), np.uint8)
    ref[oy:oy + y1 - y0, ox:ox + x1 - x0] = imgs[args.seed_image][y0:y1, x0:x1]
    ref_mask[oy:oy + y1 - y0, ox:ox + x1 - x0] = 255

    # ---- stage 1: registration, bootstrapped over three passes ------------
    # Pass 0 matches against the (cropped) seed.  Later passes match against
    # the mosaic of everything found so far, which covers the whole board and
    # is less noisy, so partial copies at the image edges get picked up too.
    lo = Warper(imgs, args.margins, (CW, CH), 1)
    for p in range(3):
        inst = find_instances(sift, ref, ref_mask, feats, full_affine=p > 0, min_inliers=args.min_inliers)
        print(f"registration pass {p}: {len(inst)} board copies")
        stack = []
        for idx, A, _, _ in inst:
            a, m = lo(idx, A)
            a[~m] = np.nan
            stack.append(a)
        stack = np.stack(stack)
        med = np.nanmedian(stack, axis=0)
        for k in range(len(stack)):
            m = np.isfinite(stack[k])
            stack[k] = np.where(m, match_exposure(np.nan_to_num(stack[k]), m, np.nan_to_num(med), np.isfinite(med)), np.nan)
        mos, _ = clipped_mean(stack)
        cover = np.isfinite(stack).any(0)
        ref, ref_mask = to_u8(mos, cover), cover.astype(np.uint8) * 255
    for idx, A, n, r in inst:
        rot = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
        print(f"  {os.path.basename(args.images[idx])}: inliers={n:4d}  residual={r:.2f}px  "
              f"scale={np.sqrt(abs(np.linalg.det(A[:, :2]))):.4f}  rot={rot:+.2f}deg  t=({A[0, 2]:.1f}, {A[1, 2]:.1f})")

    # ---- deskew: rotate the canvas frame so the board comes out square ------
    # The rotation is folded into every copy's affine, so it costs nothing:
    # the final image is still resampled from the original frames exactly once.
    seed_area = np.zeros((CH, CW), np.uint8)
    seed_area[oy:oy + y1 - y0, ox:ox + x1 - x0] = 1
    if args.deskew != "off":
        tilt, nseg = estimate_tilt(ref) if args.deskew == "auto" else (float(args.deskew), 0)
        Rm = np.vstack([cv2.getRotationMatrix2D((CW / 2, CH / 2), tilt, 1.0), [0, 0, 1]])
        inst = [(idx, (Rm @ np.vstack([A, [0, 0, 1]]))[:2], n, r) for idx, A, n, r in inst]
        seed_area = cv2.warpAffine(seed_area, Rm[:2], (CW, CH), flags=cv2.INTER_NEAREST)
        print(f"deskew: board tilted {tilt:+.2f} deg" + (f" ({nseg} line segments)" if nseg else "") + ", straightening")

    # ---- picture of what was found: every copy outlined on its frame --------
    bx0, by0, bx1, by1 = args.crop
    box = np.float32([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]])
    overlays = [cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) for im in imgs]
    for idx, A, _, _ in inst:
        corners = cv2.transform(box[None], cv2.invertAffineTransform(A))[0]
        cv2.polylines(overlays[idx], [np.round(corners).astype(np.int32)], True, (0, 120, 255), 6, cv2.LINE_AA)
    for path, ov in zip(args.images, overlays):
        cv2.imwrite(os.path.join(args.out, f"found_{os.path.splitext(os.path.basename(path))[0]}.png"), ov)

    # ---- flat-field: remove the detector's smooth brightness falloff --------
    if not args.no_flatfield:
        fields = estimate_flatfield(imgs, args.margins, inst, (CW, CH))
        imgs = [np.where(im >= 250, 255.0, np.clip(im / f, 0, 255)).astype(np.float32)
                for im, f in zip(imgs, fields)]
        f0 = np.hstack(fields)
        cv2.imwrite(os.path.join(args.out, "flatfield.png"),
                    np.clip(128 + (f0 - 1) * 128 / max(np.abs(f0 - 1).max(), 1e-6), 0, 255).astype(np.uint8))

    # ---- stage 2: parallax refinement with dense optical flow -------------
    hi = Warper(imgs, args.margins, (CW, CH), sc)
    base = [hi(idx, A) for idx, A, _, _ in inst]
    # reference view = the copy of the seed board itself (largest overlap with the seed box)
    seed_area = cv2.resize(seed_area, (hi.W, hi.H), interpolation=cv2.INTER_NEAREST) > 0
    seed = int(np.argmax([(m & seed_area).sum() if i[0] == args.seed_image else 0
                          for i, (_, m) in zip(inst, base)]))
    seed_img, seed_mask = base[seed]

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(1)
    ref, rmask = seed_img, seed_mask
    stats, shown = [], None
    centre = np.float32([[[CW / 2, CH / 2]]])
    where = [cv2.transform(centre, cv2.invertAffineTransform(A))[0, 0] for _, A, _, _ in inst]   # in its frame
    for it in range(3):
        aligned = []
        for k, ((idx, A, _, _), (b, m)) in enumerate(zip(inst, base)):
            both = m & rmask
            if k == seed or both.sum() < 20000:
                a, am = b, m
            else:
                flow = dis.calc(to_u8(ref, rmask), to_u8(b, m), None)
                flow[np.hypot(flow[..., 0], flow[..., 1]) > 6 * sc] = 0      # implausible jumps
                flow = smooth_flow(flow, cv2.erode(both.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
                a, am = hi(idx, A, flow)
                if it == 2:      # diagnostics: how large was the correction for this copy?
                    sel = both & seed_mask
                    mag = np.hypot(flow[..., 0], flow[..., 1]) / sc
                    if sel.sum() > 0.3 * seed_mask.sum():
                        med = float(np.median(mag[sel][::7]))
                        stats.append((k, med, float(np.percentile(mag[sel][::7], 95))))
                        # for the figure: the largest copy that sits far from the reference in its frame
                        far = np.hypot(*(where[k] - where[seed])) > 0.4 * min(CW, CH)
                        if far and (shown is None or sel.sum() > shown[3].sum()):
                            shown = (k, med, mag, sel)
            a = match_exposure(a, am, ref, rmask)
            a = np.where(am, a, np.nan).astype(np.float32)
            aligned.append(a)
        aligned = np.stack(aligned)
        fused, _ = clipped_mean(aligned)
        ref, rmask = fused, np.isfinite(aligned).any(0)
        print(f"parallax refinement {it + 1}/3 done")
    coverage = np.isfinite(aligned).sum(0)
    print("parallax correction per copy (vs the reference copy, source px):")
    for k, med, p95 in stats:
        d = where[k] - where[seed]
        print(f"  {os.path.basename(args.images[inst[k][0]])}: board centre {np.hypot(*d):5.0f} px from the reference's "
              f"position in the frame ({d[0]:+.0f}, {d[1]:+.0f})  ->  median {med:.2f} px, 95th pct {p95:.2f} px")

    # affine-only average, kept purely for the before/after comparison
    naive_stack = np.stack([np.where(m, match_exposure(b, m, ref, rmask), np.nan) for b, m in base]).astype(np.float32)
    naive, _ = clipped_mean(naive_stack)
    del naive_stack

    # ---- stage 3: consistency-weighted fusion -----------------------------
    # A thick part (connector, inductor) looks genuinely different from two
    # viewing angles; no 2-D warp fixes that.  So each copy is weighted, per
    # neighbourhood, by how well it agrees with one real single view R.
    R = np.where(np.isfinite(aligned[seed]), aligned[seed], fused)
    D = []
    for a in aligned:
        m = np.isfinite(a).astype(np.float32)
        d2 = np.where(m > 0, (np.nan_to_num(a) - R) ** 2, 0).astype(np.float32)
        D.append(cv2.GaussianBlur(d2, (0, 0), 3 * sc) / np.maximum(cv2.GaussianBlur(m, (0, 0), 3 * sc), 1e-3))
    D = np.stack(D)
    # Expected mismatch from noise alone.  X-ray noise depends on brightness
    # (dark = few photons = different noise), so calibrate it per grey level.
    others = [k for k in range(len(D)) if k != seed]
    Rs = cv2.GaussianBlur(R, (0, 0), 3 * sc)
    seen = np.isfinite(aligned[seed])
    lut = np.full(16, np.nan)
    for b in range(16):
        sel = seen & (Rs >= b * 16) & (Rs < b * 16 + 16)
        v = np.concatenate([D[k][sel & np.isfinite(aligned[k])][::25] for k in others])
        if len(v) > 500:
            lut[b] = np.percentile(v, 30)
    known = np.flatnonzero(np.isfinite(lut))
    lut = np.interp(np.arange(16), known, lut[known])
    noise = np.interp(Rs, np.arange(16) * 16 + 8, lut).astype(np.float32)
    # A sub-pixel registration error at a sharp edge is harmless; allow for it.
    gx, gy = cv2.Sobel(fused, cv2.CV_32F, 1, 0, ksize=3) / 8, cv2.Sobel(fused, cv2.CV_32F, 0, 1, ksize=3) / 8
    edge = (0.4 * sc) ** 2 * cv2.GaussianBlur(gx ** 2 + gy ** 2, (0, 0), 3 * sc)
    excess = np.clip((D - edge) / noise - 2.0, 0, None)
    weights = np.maximum(np.exp(-1.5 * excess), 1e-3).astype(np.float32)   # floor: never leave a hole
    weights[seed] = 1.0
    final, n_eff = clipped_mean(aligned, weights, anchor=seed)

    # ---- outputs -----------------------------------------------------------
    cx0, cy0, cx1, cy1 = [v * sc for v in args.crop]
    crop = lambda a: a[cy0:cy1, cx0:cx1]
    out = lambda name, a: cv2.imwrite(os.path.join(args.out, name), a)
    out("fused.png", crop(final).astype(np.uint8))
    out("fused_canvas.png", final.astype(np.uint8))
    out("single_view.png", crop(to_u8(aligned[seed], np.isfinite(aligned[seed]))))
    out("naive_average.png", crop(naive).astype(np.uint8))
    out("coverage.png", crop(coverage * (255 // max(int(coverage.max()), 1))).astype(np.uint8))
    out("effective_views.png", crop(np.clip(n_eff * (255.0 / len(aligned)), 0, 255)).astype(np.uint8))
    if shown is not None:     # one copy's parallax, before and after correction
        k, _, mag, sel = shown
        resid = lambda a: cv2.GaussianBlur(np.abs(np.nan_to_num(a, nan=0.0) - final) * sel, (0, 0), sc)
        before = resid(match_exposure(base[k][0], base[k][1], final, rmask))
        after = resid(aligned[k])
        top = float(np.percentile(before[sel], 99.5))
        heat = lambda a, hi_: cv2.applyColorMap(np.clip(a * (255.0 / hi_), 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        out("parallax_before.png", crop(heat(before, top)))
        out("parallax_after.png", crop(heat(after, top)))
        out("parallax_field.png", crop(heat(mag * sel, 4.0)))
        print(f"parallax figure: copy from {os.path.basename(args.images[inst[k][0]])}; "
              f"residual scale 0..{top:.0f} grey levels, field scale 0..4 px")
    c, e = crop(coverage), crop(n_eff)
    print(f"copies fused: {len(aligned)}   coverage in crop: min {c.min()} / median {int(np.median(c))} / max {c.max()}")
    print(f"effective views in crop: min {e.min():.1f} / median {np.median(e):.1f} / max {e.max():.1f}")
    print(f"wrote {os.path.join(args.out, 'fused.png')}  ({cx1 - cx0}x{cy1 - cy0})")


if __name__ == "__main__":
    main()
