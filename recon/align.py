"""Map Stage-1 target poses into the Stage-2 frame (both functions take c2w [N, 4, 4] float32 arrays)."""
import numpy as np


def align_mean_transform(ref_stage2, ref_stage1, tgt_stage1):
    """Train data: T = mean_i(P2_i @ inv(P1_i)) over refs with |t1| >= 1e-3, applied as T @ P."""
    transforms = [p2 @ np.linalg.inv(p1) for p1, p2 in zip(ref_stage1, ref_stage2)
                  if np.linalg.norm(p1[:3, 3]) >= 1e-3]
    transform = np.mean(transforms, axis=0).astype(np.float32)
    return np.array([transform @ pose for pose in tgt_stage1], dtype=np.float32)


def align_median_scale_shift(ref_stage2, ref_stage1, tgt_stage1):
    """Eval data: t' = s * t + offset with median norm ratio s and median centroids; rotations are kept."""
    valid = [i for i in range(len(ref_stage1))
             if np.linalg.norm(ref_stage1[i, :3, 3]) > 1e-3 and np.linalg.norm(ref_stage2[i, :3, 3]) > 1e-3]
    t1 = np.array([ref_stage1[i, :3, 3] for i in valid])
    t2 = np.array([ref_stage2[i, :3, 3] for i in valid])
    scale = np.median(np.linalg.norm(t2, axis=1) / np.linalg.norm(t1, axis=1))
    offset = (np.median(t2, axis=0) - scale * np.median(t1, axis=0)).astype(np.float32)
    aligned = tgt_stage1.copy()
    aligned[:, :3, 3] = float(scale) * tgt_stage1[:, :3, 3] + offset
    return aligned
