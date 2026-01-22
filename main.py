import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import cv2
import numpy as np
from tqdm import tqdm


def skew(v: np.ndarray) -> np.ndarray:
    """v: (3,) -> (3,3)"""
    x, y, z = v.reshape(-1).tolist()
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=np.float64)


def to_homo_norm(pts_px: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    pts_px: (N,2) pixel
    return: (N,3) normalized homogeneous [x,y,1]
    """
    N = pts_px.shape[0]
    pts_h = np.concatenate([pts_px, np.ones((N, 1), dtype=np.float64)], axis=1)  # (N,3)
    Kinv = np.linalg.inv(K)
    pts_n = (Kinv @ pts_h.T).T  # (N,3)
    # normalize so last coord == 1
    pts_n = pts_n / pts_n[:, 2:3]
    return pts_n


@dataclass
class TrackState:
    pt_px: np.ndarray            # (2,) current pixel
    obs: List[Tuple[int, np.ndarray]]  # list of (frame_idx, X=[x,y,1] in normalized)


def estimate_rel_R(pts1_px: np.ndarray, pts2_px: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return relative rotation R (3,3) such that x2 ~ R x1 + t
    and mask of inliers (N,1).
    """
    # OpenCV expects pixel points if you pass K
    E, inl = cv2.findEssentialMat(
        pts1_px, pts2_px, K,
        method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        raise RuntimeError("findEssentialMat failed")
    _, R, t, inl2 = cv2.recoverPose(E, pts1_px, pts2_px, K, mask=inl)
    # inl2 is refined inliers mask
    return R.astype(np.float64), inl2


def build_tracks_and_rotations(
    img_paths: List[str],
    K: np.ndarray,
    max_frames: int = 20,
    min_tracks: int = 800,
    max_corners: int = 2000,
    quality_level: float = 0.01,
    min_distance: int = 8
) -> Tuple[Dict[int, List[Tuple[int, np.ndarray]]], List[np.ndarray]]:
    """
    Use LK tracking to build multi-view tracks, and recoverPose to build chained global rotations.
    IMPORTANT IMPROVEMENT:
      - Do NOT discard lost tracks; move them to finished_tracks and keep their observations.
    Returns:
      tracks_obs: track_id -> [(frame_idx, X_norm_homo)]
      R_globals: list length T, R_globals[0]=I
    """
    img_paths = img_paths[:max_frames]
    if len(img_paths) < 2:
        raise ValueError("Need at least 2 images")

    img0 = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        raise RuntimeError(f"Failed to read {img_paths[0]}")

    # initial features
    p0 = cv2.goodFeaturesToTrack(
        img0,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance
    )
    if p0 is None:
        raise RuntimeError("goodFeaturesToTrack found nothing in the first frame")
    p0 = p0.reshape(-1, 2).astype(np.float64)

    tracks: Dict[int, TrackState] = {}
    finished_tracks: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    next_id = 0

    X0 = to_homo_norm(p0, K)
    for i in range(p0.shape[0]):
        tracks[next_id] = TrackState(pt_px=p0[i], obs=[(0, X0[i])])
        next_id += 1

    # chained global rotations (baseline; later you should replace with rotation averaging)
    R_globals: List[np.ndarray] = [np.eye(3, dtype=np.float64)]

    prev_img = img0

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    # some stats
    kept_counts = []
    lost_counts = []
    new_counts = []
    inlier_counts = []

    for f in tqdm(range(1, len(img_paths)), desc="Tracking"):
        img = cv2.imread(img_paths[f], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read {img_paths[f]}")

        ids = list(tracks.keys())

        # if no active tracks -> re-detect
        if len(ids) == 0:
            newp = cv2.goodFeaturesToTrack(
                img,
                maxCorners=max_corners,
                qualityLevel=quality_level,
                minDistance=min_distance
            )
            if newp is None:
                raise RuntimeError("No active tracks and cannot detect new points.")
            newp = newp.reshape(-1, 2).astype(np.float64)
            Xnew = to_homo_norm(newp, K)
            for i in range(newp.shape[0]):
                tracks[next_id] = TrackState(pt_px=newp[i], obs=[(f, Xnew[i])])
                next_id += 1
            ids = list(tracks.keys())

        # LK requires float32 points
        p_prev = np.array([tracks[i].pt_px for i in ids], dtype=np.float32).reshape(-1, 1, 2)

        p_next, st, err = cv2.calcOpticalFlowPyrLK(prev_img, img, p_prev, None, **lk_params)
        st = st.reshape(-1).astype(bool)
        p_next = p_next.reshape(-1, 2).astype(np.float64)

        idx_good = np.where(st)[0]
        idx_lost = np.where(~st)[0]

        good_ids = [ids[i] for i in idx_good]
        good_prev = np.array([tracks[ids[i]].pt_px for i in idx_good], dtype=np.float64)
        good_next = p_next[idx_good]

        kept_counts.append(len(good_ids))
        lost_counts.append(len(idx_lost))

        # append observations for good tracks
        Xn = to_homo_norm(good_next, K)
        for k, tid in enumerate(good_ids):
            tracks[tid].pt_px = good_next[k]
            tracks[tid].obs.append((f, Xn[k]))

        # move lost tracks to finished (do NOT discard!)
        for i in idx_lost:
            tid = ids[i]
            finished_tracks[tid] = tracks[tid].obs
            tracks.pop(tid, None)

        # ensure enough tracks: detect new points if too few
        added = 0
        if len(tracks) < min_tracks:
            newp = cv2.goodFeaturesToTrack(
                img,
                maxCorners=max_corners,
                qualityLevel=quality_level,
                minDistance=min_distance
            )
            if newp is not None:
                newp = newp.reshape(-1, 2).astype(np.float64)
                Xnew = to_homo_norm(newp, K)
                for i in range(newp.shape[0]):
                    tracks[next_id] = TrackState(pt_px=newp[i], obs=[(f, Xnew[i])])
                    next_id += 1
                    added += 1
        new_counts.append(added)

        # estimate relative rotation from correspondences between prev and current
        if good_prev.shape[0] >= 20:
            try:
                R_rel, inl = estimate_rel_R(good_prev, good_next, K)
                inlier_counts.append(int(inl.sum()))
            except Exception:
                R_rel = np.eye(3, dtype=np.float64)
                inlier_counts.append(0)
        else:
            R_rel = np.eye(3, dtype=np.float64)
            inlier_counts.append(0)

        R_globals.append(R_rel @ R_globals[-1])

        prev_img = img

    # merge active + finished tracks
    tracks_obs: Dict[int, List[Tuple[int, np.ndarray]]] = {tid: ts.obs for tid, ts in tracks.items()}
    tracks_obs.update(finished_tracks)

    # quick summary
    if len(kept_counts) > 0:
        print(f"[Track stats] kept(avg)={np.mean(kept_counts):.1f}, lost(avg)={np.mean(lost_counts):.1f}, new(avg)={np.mean(new_counts):.1f}")
        print(f"[Pose stats] inliers(avg)={np.mean(inlier_counts):.1f}")

    return tracks_obs, R_globals


def compute_u(Ri: np.ndarray, Rj: np.ndarray, Xi: np.ndarray, Xj: np.ndarray) -> float:
    """u_{i,j} = || [Xj]_x * R_{i,j} * Xi ||, where R_{i,j} = Rj * Ri^T"""
    Rij = Rj @ Ri.T
    v = skew(Xj) @ (Rij @ Xi)
    return float(np.linalg.norm(v))


def compute_aT(Ri: np.ndarray, Rj: np.ndarray, Xi: np.ndarray, Xj: np.ndarray) -> np.ndarray:
    """
    Proposition 2: a^T_{i,j} = ([R_{i,j} Xi]_x Xj)^T [Xj]_x
    returns shape (1,3)
    """
    Rij = Rj @ Ri.T
    v = skew(Rij @ Xi) @ Xj  # (3,)
    aT = (v.reshape(1, 3) @ skew(Xj))  # (1,3)
    return aT


def solve_ligt(tracks_obs: Dict[int, List[Tuple[int, np.ndarray]]],
               R_globals: List[np.ndarray],
               ref_idx: int = 0,
               min_track_len: int = 3) -> List[np.ndarray]:
    """
    Build L t = 0 using Eq.(17)-(19), with t_ref = 0 (Eq.(20)).
    Returns list of translations t_i (3,) for each frame.
    """
    T = len(R_globals)
    if not (0 <= ref_idx < T):
        raise ValueError("bad ref_idx")

    blocks: List[np.ndarray] = []

    # We'll build L over unknowns excluding ref frame => (T-1)*3 columns
    def col_slice(frame_idx: int) -> slice:
        if frame_idx == ref_idx:
            return slice(0, 0)  # unused
        # map frame index to reduced index
        ridx = frame_idx - 1 if frame_idx > ref_idx else frame_idx
        return slice(3 * ridx, 3 * ridx + 3)

    for tid, obs in tracks_obs.items():
        if len(obs) < min_track_len:
            continue

        frames = [f for (f, _) in obs]
        Xs = [X for (_, X) in obs]

        # choose (&, h) that maximizes u (Eq.(67))
        best = None
        best_u = 0.0
        for p in range(len(obs)):
            for q in range(p + 1, len(obs)):
                fi, fj = frames[p], frames[q]
                ui = compute_u(R_globals[fi], R_globals[fj], Xs[p], Xs[q])
                if ui > best_u:
                    best_u = ui
                    best = (p, q)

        if best is None or best_u < 1e-6:
            continue

        p_idx, q_idx = best
        f_amp = frames[p_idx]   # &
        f_h = frames[q_idx]     # h
        X_amp = Xs[p_idx]
        X_h = Xs[q_idx]

        # precompute u_{&;h} and a^T_{&;h}
        u_ah = compute_u(R_globals[f_amp], R_globals[f_h], X_amp, X_h)
        if u_ah < 1e-6:
            continue
        aT = compute_aT(R_globals[f_amp], R_globals[f_h], X_amp, X_h)  # (1,3)

        R_h = R_globals[f_h]
        aTRh = aT @ R_h  # (1,3)

        # For each i != & and i != h, add Eq.(17)
        for k in range(len(obs)):
            f_i = frames[k]
            if f_i == f_amp or f_i == f_h:
                continue
            X_i = Xs[k]

            R_amp_i = R_globals[f_i] @ R_globals[f_amp].T  # R_{&;i} = R_i R_&^T
            v = R_amp_i @ X_amp  # (3,)

            # B = [Xi]_x * (R_{&;i} X_& * a^T_{&;h} R_h)
            Btmp = v.reshape(3, 1) @ aTRh.reshape(1, 3)   # (3,3)
            B = skew(X_i) @ Btmp                           # (3,3)

            # C = u^2 * [Xi]_x * R_i
            C = (u_ah ** 2) * (skew(X_i) @ R_globals[f_i])  # (3,3)

            D = B + C  # (3,3)

            # Build 3 rows with blocks on t_h, t_i, t_&
            row = np.zeros((3, 3 * (T - 1)), dtype=np.float64)

            if f_h != ref_idx:
                row[:, col_slice(f_h)] += B
            if f_i != ref_idx:
                row[:, col_slice(f_i)] += C
            if f_amp != ref_idx:
                row[:, col_slice(f_amp)] += D

            blocks.append(row)

    if len(blocks) == 0:
        raise RuntimeError("No LiGT equations built. Need more/better tracks or rotations.")

    L = np.concatenate(blocks, axis=0)  # (3M, 3(T-1))

    # Solve L t = 0 => smallest singular vector
    _, _, Vt = np.linalg.svd(L, full_matrices=False)
    t_red = Vt[-1, :]  # (3(T-1),)
    t_red = t_red.reshape(-1, 3)

    # insert ref translation = 0
    t_all = []
    for i in range(T):
        if i == ref_idx:
            t_all.append(np.zeros(3, dtype=np.float64))
        else:
            ridx = i - 1 if i > ref_idx else i
            t_all.append(t_red[ridx])

    # sign disambiguation (Step 6): make a^T_{&;h} t_{&;h} >= 0 on average
    # Use a small sample of constraints
    scores = []
    for tid, obs in list(tracks_obs.items())[:200]:
        if len(obs) < 3:
            continue
        frames = [f for (f, _) in obs]
        Xs = [X for (_, X) in obs]
        # reuse same base selection as above (cheap)
        best = None
        best_u = 0.0
        for p in range(len(obs)):
            for q in range(p + 1, len(obs)):
                ui = compute_u(R_globals[frames[p]], R_globals[frames[q]], Xs[p], Xs[q])
                if ui > best_u:
                    best_u = ui
                    best = (p, q)
        if best is None or best_u < 1e-6:
            continue
        p_idx, q_idx = best
        f_amp, f_h = frames[p_idx], frames[q_idx]
        X_amp, X_h = Xs[p_idx], Xs[q_idx]
        aT = compute_aT(R_globals[f_amp], R_globals[f_h], X_amp, X_h)  # (1,3)

        t_ah = R_globals[f_h] @ (t_all[f_amp] - t_all[f_h])  # t_{&;h} = R_h (t_& - t_h)
        scores.append((aT @ t_ah.reshape(3, 1)).item())


    if len(scores) > 0 and np.median(scores) < 0:
        t_all = [-t for t in t_all]

    # normalize scale (optional)
    norms = [np.linalg.norm(t) for i, t in enumerate(t_all) if i != ref_idx]
    s = np.median(norms) if len(norms) else 1.0
    if s > 1e-9:
        t_all = [t / s for t in t_all]

    return t_all

def save_kitti_poses(path, R_globals, t_globals):
    with open(path, "w") as f:
        for R, t in zip(R_globals, t_globals):
            P = np.hstack([R, t.reshape(3,1)])  # 3x4
            f.write(" ".join(f"{v:.9e}" for v in P.reshape(-1)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", type=str, required=True, help="folder containing images")
    ap.add_argument("--max_frames", type=int, default=20)
    args = ap.parse_args()

    # TODO: 换成你的相机内参K（或自己写读取calib）
    # 示例：KITTI P0: fx 0 cx 0 / 0 fy cy 0 / 0 0 1 0
    K = np.array([[718.856, 0.0, 607.1928],
                  [0.0, 718.856, 185.2157],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    img_paths = []
    for e in exts:
        img_paths += glob.glob(os.path.join(args.img_dir, e))
    img_paths = sorted(img_paths)
    print(f"Found {len(img_paths)} images")

    tracks_obs, R_globals = build_tracks_and_rotations(img_paths, K, max_frames=args.max_frames)
    print(f"Built tracks: {len(tracks_obs)}")
    print(f"Frames: {len(R_globals)}")

    t_globals = solve_ligt(tracks_obs, R_globals, ref_idx=0)
    print("Solved translations (up to scale):")
    for i, t in enumerate(t_globals[:10]):
        print(i, t)

    save_kitti_poses("est_poses.txt", R_globals, t_globals)
    print("Saved to est_poses.txt")

    # 你现在就有 (R_i, t_i) 了：pose_i = [R_globals[i], t_globals[i]]
    # 下一步：和GT对齐（Sim(3)）算ATE/RPE；或做滑窗版本输出相对位姿。


if __name__ == "__main__":
    main()
