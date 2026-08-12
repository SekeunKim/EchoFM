"""Echo video datasets for EchoFM pretraining."""
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

def sample_clip_indices(total_frames, num_frames, frame_stride=1):
    """
    Pick num_frames indices from a clip of total_frames.

    Long clips: random contiguous window with the given stride.
    Short clips: wrap around (loop) instead of zero-padding, so the periodic
    cardiac structure the contrastive loss relies on is preserved.
    """
    if total_frames <= 0:
        raise ValueError("empty clip")
    span = (num_frames - 1) * frame_stride + 1
    if total_frames >= span:
        start = random.randint(0, total_frames - span)
        return [start + i * frame_stride for i in range(num_frames)]
    start = random.randint(0, total_frames - 1)
    return [(start + i * frame_stride) % total_frames for i in range(num_frames)]


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


class EchoDataset_from_Video_mp4(Dataset):
    def __init__(
        self,
        folder,
        image_size=[224, 224],
        channels=3,
        num_frames=32,
        frame_stride=1,
    ):
        super().__init__()
        self.folder = folder

        self.image_size = image_size
        self.channels = channels
        self.num_frames = num_frames
        self.frame_stride = frame_stride

        def create_transform(image_size):
            def transform(img):
                if not isinstance(img, Image.Image):
                    img = T.ToPILImage()(img)
                img = T.Resize(image_size)(img)
                return T.ToTensor()(img)
            return transform

        self.transform_for_videos = create_transform(self.image_size)

        self.paths = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(VIDEO_EXTS)
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"no video files found in {folder}")
        print(f"{len(self.paths)} video clips found at {folder}")

    def __len__(self):
        return len(self.paths)

    def _load(self, path):
        video = cv2.VideoCapture(path)
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            video.release()
            raise RuntimeError(f"unreadable video {path}")
        indices = sample_clip_indices(total, self.num_frames, self.frame_stride)

        # read sequentially once up to the max needed index, keep what we need
        wanted = {}
        need = set(indices)
        pos = 0
        max_idx = max(indices)
        while pos <= max_idx:
            ok, frame = video.read()
            if not ok:
                break
            if pos in need:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                wanted[pos] = self.transform_for_videos(frame)
            pos += 1
        video.release()
        if not wanted:
            raise RuntimeError(f"failed to decode frames from {path}")
        fallback = wanted[max(wanted)]
        frames = [wanted.get(i, fallback) for i in indices]
        return torch.stack(frames, dim=1)  # (C, T, H, W)

    def __getitem__(self, index):
        path = os.path.join(self.folder, self.paths[index])
        try:
            return self._load(str(path))
        except Exception as e:
            # corrupt file: fall back to a different random clip
            print(f"[EchoDataset] {e}; resampling")
            alt = random.randint(0, len(self.paths) - 1)
            return self._load(str(os.path.join(self.folder, self.paths[alt])))




class EchoDataset_from_cache_npy(Dataset):
    """
    Pretraining dataset over one or more directories of cached echo clips
    stored as .npy arrays of shape (T, H, W, 3) uint8 (BGR, as produced by
    the DICOM cache pipeline) or (T, H, W) grayscale.

    `folder` may be a single path, a comma-separated string of paths, or a
    list of paths.

    Returns float tensors (3, num_frames, image_size, image_size) in [0, 1].
    """

    def __init__(
        self,
        folder,
        num_frames=32,
        image_size=224,
        frame_stride=1,
        channel_order="bgr",
    ):
        super().__init__()
        if isinstance(folder, str):
            folders = [f for f in folder.split(",") if f]
        else:
            folders = list(folder)
        self.folders = folders
        self.num_frames = num_frames
        self.image_size = image_size
        self.frame_stride = frame_stride
        self.channel_order = channel_order

        self.paths = []
        for d in folders:
            files = sorted(
                os.path.join(d, f) for f in os.listdir(d)
                if f.endswith(".npy") or f.endswith(".npz")
            )
            print(f"{len(files)} cached clips found at {d}")
            self.paths.extend(files)
        if len(self.paths) == 0:
            raise RuntimeError(f"no .npy/.npz clips found in {folders}")
        print(f"{len(self.paths)} cached clips total")

    def __len__(self):
        return len(self.paths)

    def _load(self, path):
        if path.endswith(".npz"):
            # compressed cache (key "clip"); npz cannot be memory-mapped
            with np.load(path) as z:
                arr = z["clip"]
        else:
            arr = np.load(path, mmap_mode="r")
        if arr.ndim == 3:  # (T, H, W) grayscale -> replicate channels
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise RuntimeError(f"unexpected clip shape {arr.shape} in {path}")

        indices = sample_clip_indices(arr.shape[0], self.num_frames, self.frame_stride)
        clip = np.asarray(arr[indices])  # (num_frames, H, W, 3)
        if self.channel_order == "bgr":
            clip = clip[..., ::-1]

        tensor = torch.from_numpy(np.ascontiguousarray(clip)).float().div_(255.0)
        tensor = tensor.permute(3, 0, 1, 2)  # (3, T, H, W)

        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = F.interpolate(
                tensor,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return tensor

    def __getitem__(self, index):
        try:
            return self._load(self.paths[index])
        except Exception as e:
            print(f"[EchoDataset_npy] {e}; resampling")
            alt = random.randint(0, len(self.paths) - 1)
            return self._load(self.paths[alt])
        
         
