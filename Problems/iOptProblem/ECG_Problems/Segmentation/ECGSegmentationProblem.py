import sys
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import os
import uuid

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from trial import Point
from trial import FunctionValue
from problem import Problem
from Globalizer_problem import GlobalizerProblem

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class Sample:
    signal_path: Path
    labels_path: Path


def collect_samples(data_dir: str | Path, *, signal_suffix: str = "_signal.npy", labels_suffix: str = "_labels.npy") -> List[Sample]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"data_dir must be a directory, got: {data_dir}")

    signal_files = sorted(data_dir.glob(f"*{signal_suffix}"))
    if not signal_files:
        raise RuntimeError(f"No files '*{signal_suffix}' found in {data_dir}")

    samples: List[Sample] = []
    for s in signal_files:
        labels = s.with_name(s.name.replace(signal_suffix, labels_suffix))
        if not labels.exists():
            raise FileNotFoundError(f"Labels file not found for {s.name}: expected {labels.name}")
        samples.append(Sample(signal_path=s, labels_path=labels))
    return samples


def train_val_split(samples: Sequence[Sample], *, val_frac: float, seed: int, shuffle: bool = True) -> Tuple[List[Sample], List[Sample]]:
    if not (0.0 <= val_frac <= 1.0):
        raise ValueError(f"val_frac must be in [0,1], got {val_frac}")

    idx = np.arange(len(samples))
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(idx)

    n_val = int(len(samples) * val_frac)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train = [samples[i] for i in train_idx.tolist()]
    val = [samples[i] for i in val_idx.tolist()]
    return train, val

def train_val_test_split(
    samples: Sequence,
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
    shuffle: bool = True,
) -> Tuple[List, List, List]:
    if not (0.0 <= val_frac <= 1.0):
        raise ValueError(f"val_frac must be in [0,1], got {val_frac}")
    if not (0.0 <= test_frac <= 1.0):
        raise ValueError(f"test_frac must be in [0,1], got {test_frac}")
    if val_frac + test_frac > 1.0:
        raise ValueError(
            f"val_frac + test_frac must be <= 1, got {val_frac + test_frac}"
        )

    idx = np.arange(len(samples))
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(idx)

    n = len(samples)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)

    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]

    train = [samples[i] for i in train_idx.tolist()]
    val = [samples[i] for i in val_idx.tolist()]
    test = [samples[i] for i in test_idx.tolist()]
    return train, val, test


def infer_num_classes(samples: Sequence[Sample], *, label_channel_idx: int = 0) -> int:
    max_cls = 0
    for s in samples:
        y = np.load(s.labels_path, mmap_mode="r")
        if y.ndim == 1:
            yy = y
        elif y.ndim == 2:
            if not (0 <= label_channel_idx < y.shape[0]):
                raise ValueError(f"label_channel_idx={label_channel_idx} out of range for {s.labels_path}, shape={y.shape}")
            yy = y[label_channel_idx]
        else:
            raise ValueError(f"Unsupported labels ndim={y.ndim} in {s.labels_path} with shape={y.shape}")
        max_cls = max(max_cls, int(np.max(yy)))
    return max_cls + 1


def _to_channels_first_signal(signal: np.ndarray, labels_1d: np.ndarray, *, signal_layout: str) -> np.ndarray:
    """
    Returns signal as [C, T].
    signal_layout:
      - "channels_first": signal is [C, T] or [T] (becomes [1, T])
      - "channels_last":  signal is [T, C] (converted to [C, T]) or [T]
      - "auto": tries to infer using labels length
    """
    if signal.ndim == 1:
        return signal[None, :]

    if signal.ndim != 2:
        raise ValueError(f"signal must be 1D or 2D, got shape={signal.shape}")

    T = int(labels_1d.shape[0])

    if signal_layout == "channels_first":
        if signal.shape[1] != T:
            raise ValueError(f"channels_first expected signal [C,T], but signal.shape={signal.shape}, labels T={T}")
        return signal

    if signal_layout == "channels_last":
        if signal.shape[0] != T:
            raise ValueError(f"channels_last expected signal [T,C], but signal.shape={signal.shape}, labels T={T}")
        return np.transpose(signal, (1, 0))

    if signal_layout == "auto":
        if signal.shape[1] == T:
            return signal
        if signal.shape[0] == T:
            return np.transpose(signal, (1, 0))
        raise ValueError(f"auto layout can't match labels T={T} with signal.shape={signal.shape}")

    raise ValueError(f"Unknown signal_layout={signal_layout}")


class NpySegmentationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        *,
        normalize: bool = True,
        label_channel_idx: int = 0,
        signal_layout: str = "channels_first",
    ):
        self.samples = list(samples)
        self.normalize = normalize
        self.label_channel_idx = int(label_channel_idx)
        self.signal_layout = str(signal_layout)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        signal = np.load(s.signal_path).astype(np.float32)
        labels = np.load(s.labels_path)

        if labels.ndim == 2:
            y = labels[self.label_channel_idx].astype(np.int64)
        elif labels.ndim == 1:
            y = labels.astype(np.int64)
        else:
            raise ValueError(f"Unsupported labels shape={labels.shape} in {s.labels_path}")

        x = _to_channels_first_signal(signal, y, signal_layout=self.signal_layout)

        if x.shape[1] != y.shape[0]:
            raise ValueError(f"Length mismatch: signal {x.shape} vs labels {y.shape} in {s.signal_path}")

        if self.normalize:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-6
            x = (x - mean) / std

        return torch.from_numpy(x), torch.from_numpy(y)


def make_norm_1d(norm: str, c: int, gn_groups: int = 8) -> nn.Module:
    norm = (norm or "none").lower()
    if norm == "bn":
        return nn.BatchNorm1d(c)
    if norm == "gn":
        g = min(gn_groups, c)
        while g > 1 and c % g != 0:
            g -= 1
        return nn.GroupNorm(g, c)
    if norm == "in":
        return nn.InstanceNorm1d(c, affine=True)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm='{norm}'")


def make_act(act: str) -> nn.Module:
    act = (act or "relu").lower()
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "silu":
        return nn.SiLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    if act == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError(f"Unknown act='{act}'")


def _match_length(x: torch.Tensor, ref: torch.Tensor, pad_mode: str = "constant") -> torch.Tensor:
    diff = ref.size(-1) - x.size(-1)
    if diff > 0:
        left = diff // 2
        right = diff - left
        return F.pad(x, (left, right), mode=pad_mode) if pad_mode != "constant" else F.pad(x, (left, right))
    if diff < 0:
        diff = -diff
        left = diff // 2
        right = diff - left
        return x[..., left : x.size(-1) - right]
    return x


class ConvBlock1D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        num_convs: int = 2,
        kernel_size: int = 3,
        norm: str = "bn",
        act: str = "relu",
        dropout: float = 0.0,
        gn_groups: int = 8,
    ):
        super().__init__()
        assert num_convs >= 1
        padding = kernel_size // 2

        layers: List[nn.Module] = []
        ch = in_ch
        for _ in range(num_convs):
            layers.append(nn.Conv1d(ch, out_ch, kernel_size=kernel_size, padding=padding, bias=(norm == "none")))
            layers.append(make_norm_1d(norm, out_ch, gn_groups=gn_groups))
            layers.append(make_act(act))
            if dropout and dropout > 0:
                layers.append(nn.Dropout1d(float(dropout)))
            ch = out_ch

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock1D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        pool: str = "max",
        pool_kernel: int = 2,
        norm: str = "bn",
        act: str = "relu",
        dropout: float = 0.0,
        num_convs: int = 2,
        conv_kernel: int = 3,
        gn_groups: int = 8,
    ):
        super().__init__()
        pool = (pool or "max").lower()
        if pool == "max":
            self.down = nn.MaxPool1d(pool_kernel)
        elif pool == "avg":
            self.down = nn.AvgPool1d(pool_kernel)
        elif pool == "conv":
            k = pool_kernel
            self.down = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=2, padding=k // 2)
        else:
            raise ValueError(f"Unknown pool='{pool}'")

        self.conv = ConvBlock1D(
            in_ch=in_ch,
            out_ch=out_ch,
            num_convs=num_convs,
            kernel_size=conv_kernel,
            norm=norm,
            act=act,
            dropout=dropout,
            gn_groups=gn_groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.down(x))


class UpBlock1D(nn.Module):
    def __init__(
        self,
        in_ch_up: int,
        skip_ch: int,
        out_ch: int,
        *,
        up_mode: str = "transpose",
        norm: str = "bn",
        act: str = "relu",
        dropout: float = 0.0,
        num_convs: int = 2,
        conv_kernel: int = 3,
        gn_groups: int = 8,
        pad_mode: str = "constant",
    ):
        super().__init__()
        self.pad_mode = pad_mode
        up_mode = (up_mode or "transpose").lower()

        if up_mode == "transpose":
            self.up = nn.ConvTranspose1d(in_ch_up, out_ch, kernel_size=2, stride=2)
        elif up_mode == "interp":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                nn.Conv1d(in_ch_up, out_ch, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown up_mode='{up_mode}'")

        self.conv = ConvBlock1D(
            in_ch=out_ch + skip_ch,
            out_ch=out_ch,
            num_convs=num_convs,
            kernel_size=conv_kernel,
            norm=norm,
            act=act,
            dropout=dropout,
            gn_groups=gn_groups,
        )

    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x_up)
        x_up = _match_length(x_up, x_skip, pad_mode=self.pad_mode)
        x = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x)


class UNet1D(nn.Module):
    """
    depth = число downsample-уровней.
    depth=4 => 5 уровней (inc + 4 down), как часто делают.
    """
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        base_filters: int = 32,
        depth: int = 4,
        channel_mult: int = 2,
        max_filters: Optional[int] = None,
        num_convs_per_block: int = 2,
        conv_kernel: int = 3,
        norm: str = "bn",
        act: str = "relu",
        dropout: float = 0.0,
        gn_groups: int = 8,
        pool: str = "max",
        pool_kernel: int = 2,
        up_mode: str = "transpose",
        pad_mode: str = "constant",
        bottleneck_num_convs: Optional[int] = None,
    ):
        super().__init__()
        assert depth >= 1

        if bottleneck_num_convs is None:
            bottleneck_num_convs = num_convs_per_block

        chans: List[int] = []
        for i in range(depth + 1):
            c = base_filters * (channel_mult ** i)
            if max_filters is not None:
                c = min(c, max_filters)
            chans.append(int(c))

        self.inc = ConvBlock1D(
            in_ch=in_channels,
            out_ch=chans[0],
            num_convs=num_convs_per_block,
            kernel_size=conv_kernel,
            norm=norm,
            act=act,
            dropout=dropout,
            gn_groups=gn_groups,
        )

        self.downs = nn.ModuleList([
            DownBlock1D(
                in_ch=chans[i],
                out_ch=chans[i + 1],
                pool=pool,
                pool_kernel=pool_kernel,
                norm=norm,
                act=act,
                dropout=dropout,
                num_convs=num_convs_per_block,
                conv_kernel=conv_kernel,
                gn_groups=gn_groups,
            )
            for i in range(depth)
        ])

        self.bottleneck = ConvBlock1D(
            in_ch=chans[-1],
            out_ch=chans[-1],
            num_convs=bottleneck_num_convs,
            kernel_size=conv_kernel,
            norm=norm,
            act=act,
            dropout=dropout,
            gn_groups=gn_groups,
        )

        self.ups = nn.ModuleList([
            UpBlock1D(
                in_ch_up=chans[i + 1],
                skip_ch=chans[i],
                out_ch=chans[i],
                up_mode=up_mode,
                norm=norm,
                act=act,
                dropout=dropout,
                num_convs=num_convs_per_block,
                conv_kernel=conv_kernel,
                gn_groups=gn_groups,
                pad_mode=pad_mode,
            )
            for i in range(depth - 1, -1, -1)
        ])

        self.out_conv = nn.Conv1d(chans[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []

        x = self.inc(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)

        for i, up in enumerate(self.ups):
            skip = skips[-2 - i]
            x = up(x, skip)

        return self.out_conv(x)


class DiceCELoss(nn.Module):
    """
    logits: [B, C, T]
    target: [B, T] with class indices
    """
    def __init__(
        self,
        *,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        smooth: float = 1e-6,
        include_background: bool = False,
    ):
        super().__init__()
        self.weight_ce = float(weight_ce)
        self.weight_dice = float(weight_dice)
        self.smooth = float(smooth)
        self.include_background = bool(include_background)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, T = logits.shape
        if target.shape != (B, T):
            raise ValueError(f"Expected target [B,T], got {target.shape}")

        ce = F.cross_entropy(logits, target)

        probs = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target, num_classes=C).permute(0, 2, 1).float()

        dims = (0, 2)
        inter = (probs * target_oh).sum(dims)
        card = probs.sum(dims) + target_oh.sum(dims)
        dice_pc = (2.0 * inter + self.smooth) / (card + self.smooth)

        if self.include_background or C == 1:
            dice = dice_pc.mean()
        else:
            dice = dice_pc[1:].mean()

        dice_loss = 1.0 - dice
        return self.weight_ce * ce + self.weight_dice * dice_loss


def build_loss(name: str, *, weight_ce: float, weight_dice: float, dice_smooth: float, include_background: bool) -> nn.Module:
    name = (name or "dice_ce").lower()
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "dice_ce":
        return DiceCELoss(
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            smooth=dice_smooth,
            include_background=include_background,
        )
    raise ValueError(f"Unknown loss: {name}")


class SegmentationMeter:
    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.tp = torch.zeros(self.num_classes, dtype=torch.long)
        self.fp = torch.zeros(self.num_classes, dtype=torch.long)
        self.fn = torch.zeros(self.num_classes, dtype=torch.long)
        self.correct = 0
        self.total = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        preds = logits.argmax(dim=1).detach().cpu().reshape(-1)
        target = target.detach().cpu().reshape(-1)

        self.correct += int((preds == target).sum().item())
        self.total += int(target.numel())

        for c in range(self.num_classes):
            p = preds == c
            t = target == c
            self.tp[c] += (p & t).sum()
            self.fp[c] += (p & ~t).sum()
            self.fn[c] += (~p & t).sum()

    def accuracy(self) -> float:
        return float(self.correct) / float(max(self.total, 1))

    def f1_per_class(self, eps: float = 1e-8) -> torch.Tensor:
        tp = self.tp.float()
        fp = self.fp.float()
        fn = self.fn.float()
        denom = 2 * tp + fp + fn
        out = torch.zeros_like(tp)
        mask = denom > 0
        out[mask] = (2 * tp[mask]) / (denom[mask] + eps)
        return out

    def macro_f1(self, *, include_background: bool = False) -> float:
        f1 = self.f1_per_class()
        if self.num_classes == 1:
            return float(f1.mean().item())
        if include_background:
            vals = f1
        else:
            vals = f1[1:]
        if vals.numel() == 0:
            return 0.0
        return float(vals.mean().item())


def build_optimizer(name: str, model: nn.Module, *, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    name = (name or "adam").lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    raise ValueError(f"Unknown optimizer: {name}")


def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, criterion: nn.Module, device: str) -> Dict[str, float]:
    model.train()
    meter: Optional[SegmentationMeter] = None
    running_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()

        if meter is None:
            meter = SegmentationMeter(num_classes=int(logits.shape[1]))

        running_loss += float(loss.item()) * x.size(0)
        meter.update(logits, y)

    if meter is None:
        meter = SegmentationMeter(num_classes=1)

    return {
        "loss": running_loss / max(len(loader.dataset), 1),
        "accuracy": meter.accuracy(),
        "macro_f1": meter.macro_f1(include_background=False),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> Dict[str, float]:
    model.eval()
    meter: Optional[SegmentationMeter] = None
    running_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        if meter is None:
            meter = SegmentationMeter(num_classes=int(logits.shape[1]))

        running_loss += float(loss.item()) * x.size(0)
        meter.update(logits, y)

    if meter is None:
        meter = SegmentationMeter(num_classes=1)

    return {
        "loss": running_loss / max(len(loader.dataset), 1),
        "accuracy": meter.accuracy(),
        "macro_f1": meter.macro_f1(include_background=False),
    }





from pathlib import Path

class ECGSegmentationProblem(Problem):
    def __init__(self, dimension: int, ProcRank: int = 0):
        super(ECGSegmentationProblem, self).__init__()
        self.dimension = dimension
        self.number_of_float_variables = dimension
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        print(dimension, ProcRank)

        set_seed(int(42))

        self.model_save_path = Path("models")
        if not Path.exists(self.model_save_path):
            Path.mkdir(self.model_save_path)
        self.data_dir = Path("dataset/ECGSegmentation")
        self.signal_suffix = str("_signal.npy")
        self.labels_suffix = str("_labels.npy")
        self.samples = collect_samples(self.data_dir, signal_suffix=self.signal_suffix, labels_suffix=self.labels_suffix)
        train_s, val_s, test_s = train_val_test_split(
            self.samples,
            val_frac=0.15,
            test_frac=0.15,
            seed=42,
            shuffle=True,
        )
        self.signal_layout = "channels_first"
        self.label_channel_idx = 0
        self.normalize = True
        self.train_ds = NpySegmentationDataset(train_s, normalize=self.normalize, label_channel_idx=self.label_channel_idx,
                                          signal_layout=self.signal_layout)
        self.val_ds = NpySegmentationDataset(val_s, normalize=self.normalize, label_channel_idx=self.label_channel_idx,
                                        signal_layout=self.signal_layout)
        self.test_ds = NpySegmentationDataset(test_s, normalize=self.normalize, label_channel_idx=self.label_channel_idx,
                                        signal_layout=self.signal_layout)
        self.batch_size = 128
        self.x0, self.y0 = self.train_ds[0]
        self.in_channels = int(self.x0.shape[0])
        self.num_classes = None
        if self.num_classes is None:
            self.num_classes = infer_num_classes(self.samples, label_channel_idx=self.label_channel_idx)
        self.num_classes = int(self.num_classes)

        GPU_count = torch.cuda.device_count()
        print(f"Доступно GPU: {GPU_count}")
        for i in range(GPU_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        self.gpu_id = 0
        if GPU_count != 0:
            self.gpu_id = ProcRank % GPU_count
            
            self.device = torch.device(f"cuda:{self.gpu_id}")

            torch.device(f"{self.device}")

        else:
            self.device = "cpu"


        self.train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                  pin_memory=(str(self.device).startswith("cpu")))
        self.val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0,
                                pin_memory=(str(self.device).startswith("cpu")))
        self.test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=0,
                                pin_memory=(str(self.device).startswith("cpu")))
        self.float_variable_names = np.array(["P parameter", "Depth"], dtype=str)
        self.lower_bound_of_float_variables = [0.0, 2.0]
        self.upper_bound_of_float_variables = [1.0, 5.0]


    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        p, d = point.float_variables[0], point.float_variables[1]

        model_params = {
            'p': p,
            'd': d
        }

        model = UNet1D(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_filters=32,
            depth=int(d),
            channel_mult=2,
            max_filters=None,
            num_convs_per_block=2,
            conv_kernel=3,
            norm="bn",
            act="relu",
            dropout=p,
            gn_groups=8,
            pool="max",
            pool_kernel=2,
            up_mode="transpose",
            pad_mode="constant",
            bottleneck_num_convs=None,
        ).to(self.device)

        criterion = build_loss(
            str("dice_ce"),
            weight_ce=1.0,
            weight_dice=1.0,
            dice_smooth=1e-6,
            include_background=False,
        ).to(self.device)

        optim = build_optimizer("adamw",
            model,
            lr=1e-3,
            weight_decay=0.0,
        )

        epochs = 3
        best = -float("inf")
        best_path = None
        save_dir = "chekpoints"
        train_id = uuid.uuid4()
        for epoch in range(1, epochs + 1):
            tr = train_one_epoch(model, self.train_loader, optim, criterion, self.device)
            va = evaluate(model, self.val_loader, criterion, self.device)

            logs = {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_accuracy": tr["accuracy"],
                "train_macro_f1": tr["macro_f1"],
                "val_loss": va["loss"],
                "val_accuracy": va["accuracy"],
                "val_macro_f1": va["macro_f1"],
            }
            print(" | ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items()]))

            score = float(logs["val_macro_f1"])
            if score > best:
                best = score
                os.makedirs(save_dir, exist_ok=True)
                best_path = os.path.join(save_dir, f"{train_id}.pt")
                torch.save(model.state_dict(), best_path)

        if best_path is not None and os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=self.device))

        function_value.value = -best
        self.save_model(model, function_value.value, model_params)
        return function_value

    def save_model(self, model, metric, params_dict):
        files = [f for f in self.model_save_path.iterdir() if f.is_file()]

        for f in files:
            f_metric = float(f.stem)
            if f_metric < metric:
                return

        print("Модель сохранена: ", metric, params_dict)
        local_path = self.model_save_path.joinpath(f"{metric}.pth")
        torch.save(model.state_dict(), local_path)
        with open(self.model_save_path.joinpath(f"{metric}.json"), "w", encoding="utf-8") as f:
            json.dump(params_dict, f, ensure_ascii=False, indent=2)

    def match_by_fields(self, d1, d2, fields):
        return all(d1.get(f) == d2.get(f) for f in fields)

    def finalize(self, coordinate: List[float], discreteCoordinate: List[str]) -> float:
        print("Проверка тестовой выборки в лучшей точке: ", coordinate, discreteCoordinate)

        p, d = coordinate[0], coordinate[1]

        model_params = None

        tmp_params_dict = {
            'p': p,
            'd': d
        }

        json_files = list(Path(self.model_save_path).glob("*.json"))
        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                current_params_dict = json.load(f)
                if self.match_by_fields(current_params_dict, tmp_params_dict,
                                        ['p', 'd']):
                    model_params = current_params_dict
                    metric = float(json_file.stem)
                    break

        if model_params is None:
            print("Среди чекпоинтов не найден целевой!")
            return -1

        target_path = self.model_save_path.joinpath(f"{metric}.pth")
        print("Лучшая модель: ", target_path)

        model = UNet1D(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_filters=32,
            depth=int(model_params['d']),
            channel_mult=2,
            max_filters=None,
            num_convs_per_block=2,
            conv_kernel=3,
            norm="bn",
            act="relu",
            dropout=model_params['p'],
            gn_groups=8,
            pool="max",
            pool_kernel=2,
            up_mode="transpose",
            pad_mode="constant",
            bottleneck_num_convs=None,
        ).to(self.device)
        model.load_state_dict(torch.load(target_path, map_location=self.device))


        criterion = build_loss(
            str("dice_ce"),
            weight_ce=1.0,
            weight_dice=1.0,
            dice_smooth=1e-6,
            include_background=False,
        ).to(self.device)

        with torch.no_grad():
            test_metrics = evaluate(model, self.test_loader, criterion, self.device)

        finish_metric = test_metrics['macro_f1']
        print("Значение целевой метрики на валидационной выборке: ", metric)
        print("Значение целевой метрики на тестовой выборке: ", finish_metric)

        return finish_metric


if __name__ == '__main__':
    problem_ecg_class = ECGSegmentationProblem(2,0)
    problem = GlobalizerProblem(problem_ecg_class)
    result = problem.calculate([0.3, 4],[])
    problem.finalize([0.3, 4],[])
    print(result)