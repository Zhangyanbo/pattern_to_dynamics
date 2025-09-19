import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from tqdm.auto import tqdm


class BlobTracker:
    def __init__(
        self,
        blur_sigma: float = 1.0,  # Gaussian blur each frame for noise reduction
        min_area: float = 10.0,  # Minimum blob area (pixels)
        max_area: float = 10_000.0,  # Maximum blob area (pixels)
        min_dist: float = 8.0,  # Minimum distance between two blobs in same frame (deduplication)
        max_move: float = 30.0,  # Maximum displacement across frames (pixels)
        max_miss: int = 2,  # How many frames allowed to miss before ending trajectory
        compute_angle: bool = False,  # Whether to estimate principal axis angle (degrees)
    ):
        self.blur_sigma = blur_sigma
        self.min_area = min_area
        self.max_area = max_area
        self.min_dist = min_dist
        self.max_move = max_move
        self.max_miss = max_miss
        self.compute_angle = compute_angle

        # OpenCV SimpleBlobDetector parameters
        p = cv2.SimpleBlobDetector_Params()
        p.filterByColor = False
        p.filterByConvexity = False
        p.filterByCircularity = False
        p.filterByInertia = False

        p.filterByArea = True
        p.minArea = float(min_area)
        p.maxArea = float(max_area)

        p.minDistBetweenBlobs = float(min_dist)

        # Threshold/step determines response count, adjust as needed
        p.minThreshold = 10
        p.maxThreshold = 220
        p.thresholdStep = 10

        self.detector = cv2.SimpleBlobDetector_create(p)

    @staticmethod
    def _rgb_to_gray_u8(img3: np.ndarray) -> np.ndarray:
        # img3: (H,W,3) RGB, float or uint8
        if img3.dtype != np.uint8:
            f = img3
            if f.max() <= 1.0:
                f = f * 255.0
            img3 = np.clip(f, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)

    def _pre_blur(self, gray: np.ndarray) -> np.ndarray:
        if self.blur_sigma and self.blur_sigma > 0:
            k = max(1, int(round(self.blur_sigma * 3)) * 2 + 1)
            return cv2.GaussianBlur(gray, (k, k), self.blur_sigma)
        return gray

    def _estimate_angle(
        self, gray: np.ndarray, x: float, y: float, r: float
    ) -> Optional[float]:
        """Estimate principal axis angle (degrees) based on local second moments. r is radius; returns None if not computing."""
        if not self.compute_angle:
            return None
        H, W = gray.shape
        r = max(3, int(round(r)))
        x0, y0 = int(round(x)), int(round(y))
        x1, x2 = max(0, x0 - r), min(W, x0 + r + 1)
        y1, y2 = max(0, y0 - r), min(H, y0 + r + 1)
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        # Binarization + weighting (avoid background interference)
        thr = np.median(patch)
        mask = (patch > thr).astype(np.uint8) * 255
        m = cv2.moments(mask, binaryImage=True)
        if m["m00"] == 0:
            return None
        mu20 = m["mu20"] / m["m00"]
        mu02 = m["mu02"] / m["m00"]
        mu11 = m["mu11"] / m["m00"]
        angle = 0.5 * np.degrees(np.arctan2(2 * mu11, (mu20 - mu02 + 1e-12)))
        # Normalize to [0,180)
        if angle < 0:
            angle += 180.0
        return float(angle)

    def _detect_frame(self, gray_blur: np.ndarray) -> List[Tuple[float, float, float]]:
        """Returns [(x,y,r)], where r≈radius (from keypoint.size/2)"""
        kps = self.detector.detect(gray_blur)
        if kps is None:
            kps = []
        # Compatibility: convert tuple/list/numpy objects to list
        kps = list(kps)

        # Sort by response strength from high to low; use 0 if .response doesn't exist in some versions
        kps = sorted(kps, key=lambda k: getattr(k, "response", 0.0), reverse=True)

        detections = []
        taken = []
        for k in kps:
            # Safety: KeyPoint field names might be .pt / .size in older versions
            x, y = float(k.pt[0]), float(k.pt[1])
            r = float(max(1.0, getattr(k, "size", 0.0) / 2.0))

            # Same-frame deduplication (based on minimum distance)
            ok = True
            for xx, yy in taken:
                if (x - xx) ** 2 + (y - yy) ** 2 < (self.min_dist**2):
                    ok = False
                    break
            if ok:
                detections.append((x, y, r))
                taken.append((x, y))
        return detections

    def track(self, imgs: torch.Tensor) -> List[Dict]:
        """
        imgs: [T,1,3,H,W]  (RGB, no alpha)
        Returns: List[traj], each traj:
          {"t_start":int, "t_end":int, "corr":[(x,y),...], "angle":[...]}
        """
        assert imgs.ndim == 5 and imgs.shape[1] == 1 and imgs.shape[2] == 3
        T = imgs.shape[0]

        tracks = []  # Active trajectories
        done = []  # Completed trajectories

        def start_track(t, x, y, ang):
            tracks.append(
                {
                    "t_start": t,
                    "t_end": t,
                    "corr": [(float(x), float(y))],
                    "angle": [ang if ang is not None else None],
                    "_miss": 0,
                }
            )

        for t in tqdm(range(T), desc="Tracking"):
            frame = imgs[t, 0].detach().cpu().numpy().transpose(1, 2, 0)  # (H,W,3)
            gray = self._rgb_to_gray_u8(frame)
            gray_blur = self._pre_blur(gray)
            dets = self._detect_frame(gray_blur)  # [(x,y,r)]

            # Estimate angles (optional)
            if self.compute_angle:
                angles = [self._estimate_angle(gray, x, y, r) for (x, y, r) in dets]
            else:
                angles = [None] * len(dets)

            unmatched = set(range(len(dets)))
            used_tracks = set()

            # First associate by "old trajectory count": find nearest detection for each trajectory
            for j, tr in enumerate(tracks):
                # Find nearest among all unused detections
                best_i, best_d2 = None, 1e18
                x0, y0 = tr["corr"][-1]
                for i in list(unmatched):
                    x, y, r = dets[i]
                    d2 = (x - x0) ** 2 + (y - y0) ** 2
                    if d2 < best_d2:
                        best_d2, best_i = d2, i
                if best_i is not None and best_d2 <= (self.max_move**2):
                    x, y, r = dets[best_i]
                    tr["t_end"] = t
                    tr["corr"].append((float(x), float(y)))
                    tr["angle"].append(angles[best_i])
                    tr["_miss"] = 0
                    unmatched.discard(best_i)
                    used_tracks.add(j)
                else:
                    tr["_miss"] += 1

            # End long-term unmatched trajectories
            stay = []
            for tr in tracks:
                if tr["_miss"] > self.max_miss:
                    tr.pop("_miss", None)
                    done.append(tr)
                else:
                    stay.append(tr)
            tracks = stay

            # Remaining detections -> create new trajectories
            for i in unmatched:
                x, y, r = dets[i]
                start_track(t, x, y, angles[i])

        # Cleanup
        for tr in tracks:
            tr.pop("_miss", None)
            done.append(tr)

        return done
