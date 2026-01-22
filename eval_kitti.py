import numpy as np
import argparse
import os
import matplotlib
matplotlib.use("Agg")  # ✅ no GUI backend
import matplotlib.pyplot as plt


def load_kitti_poses(txt_path):
    poses = []
    with open(txt_path, "r") as f:
        for line in f:
            vals = line.strip().split()
            if not vals:
                continue
            if len(vals) != 12:
                raise ValueError(f"Expected 12 values per line, got {len(vals)} in {txt_path}")
            P = np.array(list(map(float, vals)), dtype=np.float64).reshape(3, 4)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = P[:, :3]
            T[:3, 3] = P[:, 3]
            poses.append(T)
    return poses


def umeyama_sim3_align(src_pts, dst_pts, with_scale=True):
    assert src_pts.shape == dst_pts.shape
    N = src_pts.shape[0]
    mu_src = src_pts.mean(axis=0)
    mu_dst = dst_pts.mean(axis=0)
    src_demean = src_pts - mu_src
    dst_demean = dst_pts - mu_dst

    cov = (dst_demean.T @ src_demean) / N
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    if with_scale:
        var_src = (src_demean ** 2).sum() / N
        scale = (D * np.diag(S)).sum() / var_src
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)
    return float(scale), R, t


def apply_sim3_to_poses(poses, s, R, t):
    out = []
    for T in poses:
        Ri = T[:3, :3]
        ti = T[:3, 3]
        Tp = np.eye(4)
        Tp[:3, :3] = R @ Ri
        Tp[:3, 3] = s * (R @ ti) + t
        out.append(Tp)
    return out


def pose_positions(poses):
    return np.array([T[:3, 3] for T in poses], dtype=np.float64)


def ate_errors(gt_poses, est_poses_aligned):
    gt_p = pose_positions(gt_poses)
    est_p = pose_positions(est_poses_aligned)
    return np.linalg.norm(est_p - gt_p, axis=1)


def relative_transform(Ta, Tb):
    return np.linalg.inv(Ta) @ Tb


def rot_angle_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def rpe_errors(gt_poses, est_poses_aligned, delta=1):
    n = min(len(gt_poses), len(est_poses_aligned))
    gt_poses = gt_poses[:n]
    est_poses_aligned = est_poses_aligned[:n]

    trans_err = []
    rot_err = []
    for i in range(n - delta):
        d_gt = relative_transform(gt_poses[i], gt_poses[i + delta])
        d_est = relative_transform(est_poses_aligned[i], est_poses_aligned[i + delta])
        E = np.linalg.inv(d_gt) @ d_est
        trans_err.append(np.linalg.norm(E[:3, 3]))
        rot_err.append(rot_angle_deg(E[:3, :3]))
    return np.array(trans_err), np.array(rot_err)


def rmse(x):
    return float(np.sqrt(np.mean(x ** 2)))


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=str, required=True)
    ap.add_argument("--est", type=str, required=True)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--delta", type=int, default=1)
    ap.add_argument("--no_scale", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_dir", type=str, default=".", help="directory to save plots")
    args = ap.parse_args()

    gt = load_kitti_poses(args.gt)
    est = load_kitti_poses(args.est)

    n = min(len(gt), len(est))
    if args.max_frames is not None:
        n = min(n, args.max_frames)
    gt = gt[:n]
    est = est[:n]

    gt_p = pose_positions(gt)
    est_p = pose_positions(est)

    s, R, t = umeyama_sim3_align(est_p, gt_p, with_scale=(not args.no_scale))
    est_aligned = apply_sim3_to_poses(est, s, R, t)

    ate = ate_errors(gt, est_aligned)
    rpe_t, rpe_r = rpe_errors(gt, est_aligned, delta=args.delta)

    print(f"Frames: {n}")
    print(f"Alignment: scale={s:.6f}  (no_scale={args.no_scale})")
    print(f"ATE: mean={ate.mean():.6f}  rmse={rmse(ate):.6f}  median={np.median(ate):.6f}")
    print(f"RPE(delta={args.delta}): trans mean={rpe_t.mean():.6f} rmse={rmse(rpe_t):.6f} | rot(deg) mean={rpe_r.mean():.6f} rmse={rmse(rpe_r):.6f}")

    if args.plot:
        os.makedirs(args.plot_dir, exist_ok=True)

        gt_p = pose_positions(gt)
        est_p_al = pose_positions(est_aligned)

        plt.figure()
        plt.plot(gt_p[:, 0], gt_p[:, 2])
        plt.plot(est_p_al[:, 0], est_p_al[:, 2])
        plt.title("Trajectory (X-Z)")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend(["GT", "EST aligned"])
        savefig(os.path.join(args.plot_dir, "traj_xz.png"))

        plt.figure()
        plt.plot(ate)
        plt.title("ATE per frame")
        plt.xlabel("frame")
        plt.ylabel("ATE (m)")
        savefig(os.path.join(args.plot_dir, "ate.png"))

        plt.figure()
        plt.plot(rpe_t)
        plt.title(f"RPE translation (delta={args.delta})")
        plt.xlabel("frame")
        plt.ylabel("trans error (m)")
        savefig(os.path.join(args.plot_dir, "rpe_trans.png"))

        plt.figure()
        plt.plot(rpe_r)
        plt.title(f"RPE rotation (delta={args.delta})")
        plt.xlabel("frame")
        plt.ylabel("rot error (deg)")
        savefig(os.path.join(args.plot_dir, "rpe_rot.png"))

        print(f"Saved plots to: {os.path.abspath(args.plot_dir)}")


if __name__ == "__main__":
    main()
