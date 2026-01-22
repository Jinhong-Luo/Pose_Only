import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class Match:
    track_id: int
    pt_prev: Tuple[float, float]
    pt_curr: Tuple[float, float]
    status: int          # 1=kept, 0=filtered out
    reason: str = ""     # "", "lk_fail", "fb_fail", "ransac_outlier"


@dataclass
class KLTConfig:
    # --- Feature detection (GFTT / Shi-Tomasi) ---
    max_corners: int = 2000
    quality_level: float = 0.01
    min_distance: int = 8
    block_size: int = 3
    use_harris: bool = False
    k: float = 0.04

    # --- Tracking (PyrLK) ---
    win_size: Tuple[int, int] = (21, 21)
    max_level: int = 3
    criteria: Tuple[int, int, float] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    # --- Quality checks ---
    fb_thresh: float = 1.0              # forward-backward pixel error threshold
    ransac_reproj_thresh: float = 1.0   # for findFundamentalMat
    ransac_conf: float = 0.999

    # --- Track management ---
    min_tracks: int = 800               # if alive tracks < min_tracks => detect new points
    border: int = 8                     # avoid detecting points near image border
    keep_history: bool = True           # store per-track history
    max_history: int = 200              # truncate history length


class KLTTracker:
    """
    Minimal KLT tracker:
    - Maintains track IDs
    - Outputs pairwise matches (prev->curr) + current alive tracks
    """

    def __init__(self, cfg: KLTConfig = KLTConfig()):
        self.cfg = cfg
        self.prev_img: Optional[np.ndarray] = None

        # track_id -> (x, y)
        self.tracks: Dict[int, Tuple[float, float]] = {}
        self.next_id: int = 0

        # optional: track_id -> list[(frame_idx, x, y)]
        self.history: Dict[int, List[Tuple[int, float, float]]] = {}
        self.frame_idx: int = -1

    def reset(self):
        self.prev_img = None
        self.tracks.clear()
        self.history.clear()
        self.next_id = 0
        self.frame_idx = -1

    def _detect_new_points(self, img: np.ndarray, existing_pts: np.ndarray) -> np.ndarray:
        """
        Detect new GFTT corners, avoiding regions near existing points and near borders.
        Returns Nx2 float array.
        """
        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # suppress border
        b = self.cfg.border
        mask[:b, :] = 0
        mask[-b:, :] = 0
        mask[:, :b] = 0
        mask[:, -b:] = 0

        # suppress around existing points
        if existing_pts is not None and len(existing_pts) > 0:
            for x, y in existing_pts.reshape(-1, 2):
                cv2.circle(mask, (int(x), int(y)), self.cfg.min_distance, 0, -1)

        pts = cv2.goodFeaturesToTrack(
            img,
            maxCorners=self.cfg.max_corners,
            qualityLevel=self.cfg.quality_level,
            minDistance=self.cfg.min_distance,
            mask=mask,
            blockSize=self.cfg.block_size,
            useHarrisDetector=self.cfg.use_harris,
            k=self.cfg.k,
        )
        if pts is None:
            return np.zeros((0, 2), dtype=np.float32)
        return pts.reshape(-1, 2).astype(np.float32)

    def _append_history(self, tid: int, x: float, y: float):
        if not self.cfg.keep_history:
            return
        lst = self.history.setdefault(tid, [])
        lst.append((self.frame_idx, float(x), float(y)))
        if len(lst) > self.cfg.max_history:
            del lst[:-self.cfg.max_history]

    def update(self, curr_img: np.ndarray) -> Dict[str, Any]:
        """
        Args:
            curr_img: grayscale uint8 image (H,W)

        Returns dict:
            - frame_idx: int
            - matches: List[Match]
            - tracks: Dict[int, (x,y)]   alive tracks at curr frame
            - history: optional Dict[int, List[(frame_idx,x,y)]]
        """
        if curr_img.ndim != 2 or curr_img.dtype != np.uint8:
            raise ValueError("curr_img must be grayscale uint8 image (H,W), dtype uint8")

        self.frame_idx += 1

        # First frame: just detect points and init tracks
        if self.prev_img is None:
            pts = self._detect_new_points(curr_img, existing_pts=None)
            for (x, y) in pts:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = (float(x), float(y))
                self._append_history(tid, x, y)

            self.prev_img = curr_img
            return {
                "frame_idx": self.frame_idx,
                "matches": [],              # no previous frame
                "tracks": dict(self.tracks),
                "history": dict(self.history) if self.cfg.keep_history else None,
            }

        # If no tracks, treat as re-init
        if len(self.tracks) == 0:
            self.prev_img = None
            return self.update(curr_img)

        # Prepare previous points array
        prev_ids = np.array(list(self.tracks.keys()), dtype=np.int64)
        prev_pts = np.array([self.tracks[tid] for tid in prev_ids], dtype=np.float32).reshape(-1, 1, 2)

        # --- Forward LK: prev -> curr ---
        curr_pts, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_img, curr_img, prev_pts, None,
            winSize=self.cfg.win_size,
            maxLevel=self.cfg.max_level,
            criteria=self.cfg.criteria
        )
        st = st.reshape(-1).astype(bool)
        curr_pts = curr_pts.reshape(-1, 2)

        # Build initial matches, mark LK failures
        matches: List[Match] = []
        for i, tid in enumerate(prev_ids):
            if not st[i]:
                matches.append(Match(int(tid), tuple(prev_pts[i, 0]), (np.nan, np.nan), 0, "lk_fail"))
            else:
                matches.append(Match(int(tid), tuple(prev_pts[i, 0]), tuple(curr_pts[i]), 1, ""))

        # Keep only LK-success for next checks
        good_mask = st.copy()
        good_prev = prev_pts.reshape(-1, 2)[good_mask]
        good_curr = curr_pts[good_mask]
        good_ids = prev_ids[good_mask]

        # --- Forward-Backward check ---
        if len(good_prev) >= 1:
            back_pts, st_back, _ = cv2.calcOpticalFlowPyrLK(
                curr_img, self.prev_img,
                good_curr.reshape(-1, 1, 2), None,
                winSize=self.cfg.win_size,
                maxLevel=self.cfg.max_level,
                criteria=self.cfg.criteria
            )
            st_back = st_back.reshape(-1).astype(bool)
            back_pts = back_pts.reshape(-1, 2)

            fb_err = np.linalg.norm(back_pts - good_prev, axis=1)
            fb_ok = (st_back & (fb_err < self.cfg.fb_thresh))

            # mark failures in matches list
            # map good_ids index -> global matches index
            id_to_match_idx = {int(tid): idx for idx, tid in enumerate(prev_ids)}
            for i, tid in enumerate(good_ids):
                if not fb_ok[i]:
                    mi = id_to_match_idx[int(tid)]
                    matches[mi].status = 0
                    matches[mi].reason = "fb_fail"

            # filter
            good_prev = good_prev[fb_ok]
            good_curr = good_curr[fb_ok]
            good_ids = good_ids[fb_ok]

        # --- RANSAC geometric filtering (Fundamental matrix) ---
        if len(good_prev) >= 8:
            F, inlier_mask = cv2.findFundamentalMat(
                good_prev, good_curr,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=self.cfg.ransac_reproj_thresh,
                confidence=self.cfg.ransac_conf
            )
            if inlier_mask is not None:
                inlier_mask = inlier_mask.reshape(-1).astype(bool)
            else:
                inlier_mask = np.zeros((len(good_prev),), dtype=bool)

            id_to_match_idx = {int(tid): idx for idx, tid in enumerate(prev_ids)}
            for i, tid in enumerate(good_ids):
                if not inlier_mask[i]:
                    mi = id_to_match_idx[int(tid)]
                    matches[mi].status = 0
                    matches[mi].reason = "ransac_outlier"

            good_prev = good_prev[inlier_mask]
            good_curr = good_curr[inlier_mask]
            good_ids = good_ids[inlier_mask]

        # --- Update alive tracks using filtered correspondences ---
        new_tracks: Dict[int, Tuple[float, float]] = {}
        for tid, (x, y) in zip(good_ids, good_curr):
            new_tracks[int(tid)] = (float(x), float(y))

        self.tracks = new_tracks

        # --- Detect & add new points if needed ---
        if len(self.tracks) < self.cfg.min_tracks:
            existing = np.array(list(self.tracks.values()), dtype=np.float32)
            new_pts = self._detect_new_points(curr_img, existing_pts=existing if len(existing) > 0 else None)

            # add until max_corners total (soft cap)
            # (you can change this behavior as needed)
            need = max(0, self.cfg.max_corners - len(self.tracks))
            if need > 0 and len(new_pts) > 0:
                new_pts = new_pts[:need]
                for (x, y) in new_pts:
                    tid = self.next_id
                    self.next_id += 1
                    self.tracks[tid] = (float(x), float(y))

        # --- Update history ---
        if self.cfg.keep_history:
            for tid, (x, y) in self.tracks.items():
                self._append_history(tid, x, y)

        # shift
        self.prev_img = curr_img

        return {
            "frame_idx": self.frame_idx,
            "matches": matches,
            "tracks": dict(self.tracks),
            "history": dict(self.history) if self.cfg.keep_history else None,
        }


# ------------------------ Example usage (KITTI image_0) ------------------------
if __name__ == "__main__":
    import glob
    from pathlib import Path

    # Change this to your KITTI sequence path:
    # e.g. /path/to/KITTI/odometry/dataset/sequences/00/image_0/*.png
    img_glob = "D:/Program/PythonProject/Pose_only/dataset/00/test/*.png"
    imgs = sorted(glob.glob(img_glob))

    tracker = KLTTracker(KLTConfig(min_tracks=800, max_corners=2000))

    for p in imgs[:200]:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        out = tracker.update(img)

        matches = out["matches"]
        alive = out["tracks"]

        # pairwise matches between previous and current:
        good_pairs = [(m.track_id, m.pt_prev, m.pt_curr) for m in matches if m.status == 1]
        print(Path(p).name, "alive:", len(alive), "pair_matches:", len(good_pairs))
