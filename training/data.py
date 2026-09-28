import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from recon.scenes import TRAIN_DATASETS, find_scenes, frame_paths, load_images


class SplatGuideDataset(Dataset):
    """Samples of 21 views drawn on the fly from raw multi-view image datasets (references first).
    WorldMirror, renders, latents and CLIP are computed on the GPU (see training/engine.py)."""

    def __init__(self, datasets: list[dict]):
        """datasets: [{name: dl3dv | re10k, root: ..., repeat: k (optional)}]"""
        self.scenes = []
        for d in datasets:
            spec = TRAIN_DATASETS[d["name"]]
            low, high = spec["num_frames"]
            scenes = [
                s for s in find_scenes(d["root"])
                if low <= len(frame_paths(s, spec["use_transforms_json"])) <= high
            ]
            print(f"Found {len(scenes)} usable {d['name']} scenes in {d['root']}")
            self.scenes += [(d["name"], s) for s in scenes] * d.get("repeat", 1)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        name, scene = self.scenes[idx]
        spec = TRAIN_DATASETS[name]
        frames = frame_paths(scene, spec["use_transforms_json"])
        ref_ids, tgt_ids = spec["sampler"](len(frames), np.random)
        try:
            images_wm, images = load_images([frames[i] for i in ref_ids + tgt_ids])
        except OSError as e:  # unreadable image: draw another scene
            print(f"[WARNING] {scene}: {e}")
            return self[np.random.randint(len(self))]
        return {"images_wm": images_wm, "images": images, "n_ref": len(ref_ids)}


def first(batch: list):
    return batch[0]


class SplatGuideDataModule(LightningDataModule):
    def __init__(self, datasets: list[dict], num_workers: int = 4):
        super().__init__()
        self.dataset = SplatGuideDataset(datasets)
        self.num_workers = num_workers

    def train_dataloader(self):
        # One sample per GPU and step; Lightning adds the DistributedSampler.
        return DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=first,
            num_workers=self.num_workers,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
        )
