"""Raw multi-view datasets: scene discovery, frame lists, image loading and view selection
(random train sampling / eval split files)."""
import json
import os

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
IMAGE_SUBDIRS = ("images", "images_4", "images_2", "images_8", "images_1")
NUM_TRAIN_VIEWS = 21
WM_SIZE = 448  # WorldMirror input resolution
OUT_SIZE = 576  # render / diffusion resolution


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))


def image_dir(scene):
    """DL3DV / Mip-NeRF 360 keep frames in images*/; RE10K scenes hold them directly."""
    return next((os.path.join(scene, d) for d in IMAGE_SUBDIRS if os.path.isdir(os.path.join(scene, d))), scene)


def is_scene(path):
    return os.path.exists(os.path.join(path, "transforms.json")) or len(list_images(image_dir(path))) > 0


def find_scenes(root):
    """Scenes directly under root, or under root/<category>/ (e.g. DL3DV 1K, 2K, ...)."""
    subdirs = sorted(d for d in os.listdir(root) if not d.startswith(".") and os.path.isdir(os.path.join(root, d)))
    if subdirs and is_scene(os.path.join(root, subdirs[0])):
        return [os.path.join(root, d) for d in subdirs if is_scene(os.path.join(root, d))]
    return [os.path.join(root, c, s) for c in subdirs for s in sorted(os.listdir(os.path.join(root, c)))
            if not s.startswith(".") and os.path.isdir(os.path.join(root, c, s)) and is_scene(os.path.join(root, c, s))]


def frame_paths(scene, use_transforms_json=False):
    """All frames of a scene in index order (DL3DV train data follows transforms.json, the rest sorted names)."""
    folder = image_dir(scene)
    if use_transforms_json:
        with open(os.path.join(scene, "transforms.json")) as f:
            names = [os.path.basename(frame["file_path"]) for frame in json.load(f)["frames"]]
    else:
        names = list_images(folder)
    return [os.path.join(folder, n) for n in names]


def sample_views_dl3dv(n_frames, rng):
    """21 random frames; frame 0 of those plus 1-11 random others are refs."""
    selected = np.sort(rng.choice(n_frames, NUM_TRAIN_VIEWS, replace=False))
    n_ref = rng.randint(2, 13)
    ref = [0] + sorted(rng.choice(np.arange(1, NUM_TRAIN_VIEWS), n_ref - 1, replace=False).tolist())
    return split_selected(selected, ref)


def sample_views_re10k(n_frames, rng):
    """21 frames (strided for long clips, else random) excluding the last 5;
    2-5 refs with one in the first and one in the last 30% of the 21."""
    usable = n_frames - 5
    if usable > 150 and rng.rand() < 0.6:
        stride = 5 if usable <= 200 else 8  # always yields >= 21 frames
        selected = np.arange(rng.randint(0, stride * 2), usable, stride)[:NUM_TRAIN_VIEWS]
    else:
        selected = np.sort(rng.choice(np.arange(usable), NUM_TRAIN_VIEWS, replace=False))

    n_ref = rng.randint(2, 6)
    ref = [rng.choice(7), rng.choice(np.arange(14, NUM_TRAIN_VIEWS))]  # first / last 30% of the 21
    ref += rng.choice([i for i in range(NUM_TRAIN_VIEWS) if i not in ref], n_ref - 2, replace=False).tolist()
    return split_selected(selected, sorted(ref))


def split_selected(selected, ref):
    """Map positions within the 21 selected frames to (ref, target) frame indices."""
    return [int(selected[i]) for i in ref], [int(selected[i]) for i in range(len(selected)) if i not in ref]


# Per training dataset: view sampler, frame order from transforms.json, usable clip lengths.
TRAIN_DATASETS = {
    "dl3dv": {"sampler": sample_views_dl3dv, "use_transforms_json": True, "num_frames": (NUM_TRAIN_VIEWS, 10**9)},
    "re10k": {"sampler": sample_views_re10k, "use_transforms_json": False, "num_frames": (65, 280)},
}


def split_file(split_dir, scene_name, split_num):
    return os.path.join(split_dir, scene_name, f"train_test_split_{split_num}.json")


def eval_views(split_dir, scene_name, split_num):
    with open(split_file(split_dir, scene_name, split_num)) as f:
        split = json.load(f)
    return split["train_ids"], split["test_ids"]


def load_images(paths, sizes=(WM_SIZE, OUT_SIZE)):
    """Center square crop + LANCZOS resize, decoding each image once.
    Returns one [N, 3, size, size] tensor in [0, 1] per size."""
    images = [[] for _ in sizes]
    for path in paths:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
        for out, size in zip(images, sizes):
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
            out.append(torch.from_numpy(np.array(resized).astype(np.float32) / 255.0).permute(2, 0, 1))
    return [torch.stack(out) for out in images]
