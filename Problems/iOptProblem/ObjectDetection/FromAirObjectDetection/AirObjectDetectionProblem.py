import torch.nn.functional as F
import cv2
import albumentations as A
import math
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import torch
import torch.nn as nn
from typing import List
import sys
import json

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from trial import Point
from trial import FunctionValue
from problem import Problem
from Globalizer_problem import GlobalizerProblem

"""# Подготовка"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется: {device}")

base_dir = "./dataset/FromAirObjectDetection"
images_dir = os.path.join(base_dir, "color")
masks_dir = os.path.join(base_dir, "mask")

all_files = sorted([
  os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(
    (".png", ".jpg", ".jpeg")
  )
])

train_val_files, test_files = train_test_split(all_files, test_size = 0.1,
                                               random_state = 42)
train_files, val_files = train_test_split(train_val_files, test_size = 0.222,
                                          random_state = 42)


NUM_CLASSES = 12
BATCH_SIZE = 16
NUM_WORKERS = os.cpu_count() if os.name != "nt" else 0
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok = True)
EPOCHS_DIR = os.path.join(CHECKPOINT_DIR, "epochs")
os.makedirs(EPOCHS_DIR, exist_ok = True)

ITERS_LIMIT = 200 
EPOCHS_FOR_OPT = 3
R_OPT = 3.0 
LOG10_LR_MIN_OPT = -4.0
LOG10_LR_MAX_OPT = -2.0

EPOCHS_FOR_TRAIN = 200 
BEST_MODEL_NAME = "MODEL_BEST.pth"

# Параметры изображений датасета
IMAGES_EXTENSIONS = [".png", ".jpg", ".jpeg"]
MASK_EXTENSION = ".png"
IMAGE_SIZE = 256

"""# Цвета"""

COLOR_MAP = {
  "background":   [0, 0, 0],
  "person":       [192, 128, 128],
  "bike":         [0, 128, 0],
  "car":          [128, 128, 128],
  "drone":        [128, 0, 0],
  "boat":         [0, 0, 128],
  "animal":       [192, 0, 128],
  "obstacle":     [192, 0, 0],
  "construction": [192, 128, 0],
  "vegetation":   [0, 64, 0],
  "road":         [128, 128, 0],
  "sky":          [0, 128, 128]
}

CLASSES = list(COLOR_MAP.keys())
COLOR_LIST = np.array(list(COLOR_MAP.values()))

"""# Преобразования цветов"""

def rgb_to_mask(rgb_mask):
  h, w = rgb_mask.shape[:2]
  mask = np.zeros((h, w), dtype = np.int64)
  for i, color in enumerate(COLOR_LIST):
    match = np.all(rgb_mask == color, axis = -1)
    mask[match] = i
  return mask

def mask_to_rgb(mask_indices):
  h, w = mask_indices.shape
  rgb = np.zeros((h, w, 3), dtype = np.uint8)
  for i, color in enumerate(COLOR_LIST):
    rgb[mask_indices == i] = color
  return rgb

"""# Датасет"""

class DroneDataset(Dataset):
  def __init__(self, filenames, images_dir, masks_dir, transform = None):
    self.filenames = filenames
    self.images_dir = images_dir
    self.masks_dir = masks_dir
    self.transform = transform

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    base_name = self.filenames[idx]

    img_path = None
    for ext in IMAGES_EXTENSIONS:
      p = os.path.join(self.images_dir, base_name + ext)
      if os.path.exists(p):
        img_path = p
        break

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_path = os.path.join(self.masks_dir, base_name + MASK_EXTENSION)
    mask_rgb = cv2.imread(mask_path)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)

    mask_indices = rgb_to_mask(mask_rgb)

    if self.transform:
      augmented = self.transform(image = image, mask = mask_indices)
      image = augmented["image"]
      mask_indices = augmented["mask"]

    if not isinstance(mask_indices, torch.Tensor):
      mask_indices = torch.from_numpy(mask_indices)

    return image, mask_indices.long()

"""# Трансформации"""

train_transform = A.Compose([
  A.Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation = cv2.INTER_LINEAR),
  A.HorizontalFlip(p = 0.5),
  A.Affine(rotate = (-15, 15), p = 0.5),
  A.RandomBrightnessContrast(p = 0.2),
  A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
  ToTensorV2()
])

val_test_transform = A.Compose([
  A.Resize(IMAGE_SIZE, IMAGE_SIZE),
  A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
  ToTensorV2()
])


class CNA(nn.Module):
  def __init__(self, in_nc, out_nc, stride = 1):
    super().__init__()

    self.conv = nn.Conv2d(in_nc, out_nc, 3, stride = stride,
                          padding = 1, bias = False)
    self.norm = nn.BatchNorm2d(out_nc)
    self.act = nn.GELU()

  def forward(self, x):
    out = self.conv(x)
    out = self.norm(out)
    out = self.act(out)

    return out

class UnetBlock(nn.Module):
  def __init__(self, in_nc, inner_nc, out_nc, inner_block = None):
    super().__init__()

    self.conv1 = CNA(in_nc, inner_nc, stride = 2)
    self.conv2 = CNA(inner_nc, inner_nc)
    self.inner_block = inner_block
    self.conv3 = CNA(inner_nc, inner_nc)
    self.conv_cat = nn.Conv2d(inner_nc + in_nc, out_nc, 3, padding = 1)

  def forward(self, x):
    _, _, h, w = x.shape

    inner = self.conv1(x)
    inner = self.conv2(inner)

    if self.inner_block is not None:
      inner = self.inner_block(inner)
    inner = self.conv3(inner)

    inner = F.interpolate(inner, size = (h, w), mode = "bilinear")
    inner = torch.cat((x, inner), axis = 1)
    out = self.conv_cat(inner)

    return out

class Unet(nn.Module):
  def __init__(self, in_nc = 3, nc = 32, out_nc = NUM_CLASSES, num_downs = 5):
    super().__init__()

    self.cna1 = CNA(in_nc, nc)
    self.cna2 = CNA(nc, nc)

    unet_block = None
    for i in range(num_downs - 3):
      unet_block = UnetBlock(8 * nc, 8 * nc, 8 * nc, unet_block)
    unet_block = UnetBlock(4 * nc, 8 * nc, 4 * nc, unet_block)
    unet_block = UnetBlock(2 * nc, 4 * nc, 2 * nc, unet_block)
    self.unet_block = UnetBlock(nc, 2 * nc, nc, unet_block)

    self.cna3 = CNA(nc, nc)
    self.conv_last = nn.Conv2d(nc, out_nc, 3, padding = 1)

  def forward(self, x):
    out = self.cna1(x)
    out = self.cna2(out)
    out = self.unet_block(out)
    out = self.cna3(out)
    out = self.conv_last(out)

    return out

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DiceLoss(nn.Module):
  def __init__(self, smooth=1e-6):
    super().__init__()
    self.smooth = smooth

  def forward(self, preds, targets):
    # preds: [batch, classes, H, W]
    # targets: [batch, H, W]

    # Преобразуем в one-hot
    preds = torch.softmax(preds, dim=1)
    targets_one_hot = torch.eye(preds.shape[1], device=preds.device)[targets]
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)

    # Dice для каждого класса
    intersection = (preds * targets_one_hot).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
    dice = (2. * intersection + self.smooth) / (union + self.smooth)

    # Усредняем по классам и батчу
    return 1 - dice.mean()

class CombinedLoss(nn.Module):
  def __init__(self, ce_weight=0.5, dice_weight=0.5):
    super().__init__()
    self.ce_loss = nn.CrossEntropyLoss()
    self.dice_loss = DiceLoss()
    self.ce_weight = ce_weight
    self.dice_weight = dice_weight

  def forward(self, preds, targets):
    ce = self.ce_loss(preds, targets)
    dice = self.dice_loss(preds, targets)
    return self.ce_weight * ce + self.dice_weight * dice

class AirObjectDetectionProblem(Problem):
  def __init__(self, dimension: int, ProcRank:int=0):
    super(AirObjectDetectionProblem, self).__init__()

    self.train_val_files, self.test_files = train_test_split(all_files, test_size=0.15,
                                                   random_state=42)

    self.train_files, self.val_files = train_test_split(self.train_val_files, test_size=0.15,
                                                   random_state=42)


    self.test_ds = DroneDataset(self.test_files, images_dir, masks_dir,
                           transform = val_test_transform)
    self.train_ds = DroneDataset(self.train_files, images_dir, masks_dir,
                           transform = val_test_transform)
    self.val_ds = DroneDataset(self.val_files, images_dir, masks_dir,
                           transform = val_test_transform)

    self.train_loader = DataLoader(self.train_ds, batch_size = BATCH_SIZE,
                              shuffle = True, num_workers = NUM_WORKERS)
    self.val_loader = DataLoader(self.val_ds, batch_size = BATCH_SIZE,
                            shuffle = False, num_workers = NUM_WORKERS)
    self.test_loader = DataLoader(self.test_ds, batch_size = 1, shuffle = False)

    self.model_save_path = Path("models")
    if not Path.exists(self.model_save_path):
        Path.mkdir(self.model_save_path)


    self.dimension = dimension
    self.number_of_float_variables = 1
    self.number_of_discrete_variables = 2
    self.number_of_objectives = 1
    self.number_of_constraints = 0

    self.float_variable_names = np.array(["log10_learning_rate"], dtype = str)
    self.lower_bound_of_float_variables = np.array([LOG10_LR_MIN_OPT],
                                                   dtype = np.double)
    self.upper_bound_of_float_variables = np.array([LOG10_LR_MAX_OPT],
                                                   dtype = np.double)

    self.discrete_variable_names = ["base_nc", "num_downs"]
    self.discrete_variable_values = [
        ["16", "32", "64", "128"],
        ["4", "5", "6", "7"]
    ]

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

    print(f"gpu_id = {self.gpu_id}")

  def calculate_iou_dice(self, preds, masks):
    """
    Вычисляет IoU и Dice коэффициент для батча
    """
    # Получаем предсказанные классы (индексы с максимальной вероятностью)
    preds_classes = torch.argmax(preds, dim=1)  # shape: [batch_size, H, W]

    batch_iou = 0
    batch_dice = 0

    for i in range(preds_classes.shape[0]):
      pred = preds_classes[i]  # [H, W]
      mask = masks[i]  # [H, W]

      # Считаем пересечение и объединение
      intersection = (pred == mask).float().sum()
      union = (pred + mask > 0).float().sum()
      iou = intersection / (union + 1e-6)
      dice = 2 * intersection / ((pred == mask).float().sum() + (mask == mask).float().sum() + 1e-6)
      batch_iou += iou
      batch_dice += dice

    return batch_iou / preds_classes.shape[0], batch_dice / preds_classes.shape[0]

  def calculate(self, point: Point,
                function_value: FunctionValue) -> FunctionValue:
    lr = 10 ** point.float_variables[0]
    base_nc = int(point.discrete_variables[0])
    num_downs = int(point.discrete_variables[1])

    model_params = {
        'base_nc': base_nc,
        'num_downs': num_downs,
        'lr': lr,
    }

    print(f"\n[iOpt] Testing: LR = {lr:.5f}, NC = {base_nc}, Downs = {num_downs}")

    model = Unet(in_nc = 3, nc = base_nc, out_nc = NUM_CLASSES,
                 num_downs = num_downs).to(self.device)

    criterion = CombinedLoss(ce_weight=0.3, dice_weight=0.7)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    s = 0
    for epoch in range(EPOCHS_FOR_OPT):
      model.train()

      for imgs, masks in self.train_loader:
        imgs = imgs.to(self.device)
        masks = masks.to(self.device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        if not math.isnan(loss.item()):
          optimizer.step()

    model.eval()
    val_loss = 0
    val_iou = 0
    val_dice = 0
    valid_batches = 0

    with torch.no_grad():
      for imgs, masks in self.val_loader:
        imgs = imgs.to(self.device)
        masks = masks.to(self.device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        iou, dice = self.calculate_iou_dice(preds, masks)
        if not math.isnan(loss.item()):
          val_loss += loss.item()
          val_iou += iou
          val_dice += dice
          valid_batches += 1

    if valid_batches == 0:
      avg_val_loss = 999.0
      avg_val_iou = 0.0
      avg_val_dice = 0.0
    else:
      avg_val_loss = val_loss / valid_batches
      avg_val_iou = val_iou / valid_batches
      avg_val_dice = val_dice / valid_batches
    print(f"Result: Loss={avg_val_loss:.4f}, IoU={avg_val_iou:.4f}, Dice={avg_val_dice:.4f}")
    f1_score = 2 * avg_val_iou * avg_val_dice / (avg_val_iou + avg_val_dice + 1e-6)
    function_value.value = (-1) * f1_score
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
      lr = 10 ** coordinate[0]
      base_nc = int(discreteCoordinate[0])
      num_downs = int(discreteCoordinate[1])

      model_params = None

      tmp_params_dict = {
          'base_nc': base_nc,
          'num_downs': num_downs,
          'lr': lr,
      }

      json_files = list(Path(self.model_save_path).glob("*.json"))
      for json_file in json_files:
          with open(json_file, "r", encoding="utf-8") as f:
              current_params_dict = json.load(f)
              if self.match_by_fields(current_params_dict, tmp_params_dict,
                                      ['base_nc', 'num_downs', 'lr']):
                  model_params = current_params_dict
                  metric = float(json_file.stem)
                  break

      if model_params is None:
          print("Среди чекпоинтов не найден целевой!")
          return -1

      target_path = self.model_save_path.joinpath(f"{metric}.pth")
      print("Лучшая модель: ", target_path)

      model = Unet(in_nc=3, nc=int(model_params['base_nc']), out_nc=NUM_CLASSES,
                   num_downs=int(model_params['num_downs'])).to(self.device)

      model.load_state_dict(torch.load(target_path, map_location=self.device))
      model.eval()

      val_loss = 0
      val_iou = 0
      val_dice = 0
      valid_batches = 0
      criterion = CombinedLoss(ce_weight=0.3, dice_weight=0.7)

      with torch.no_grad():
          for imgs, masks in self.test_loader:
              imgs = imgs.to(self.device)
              masks = masks.to(self.device)
              preds = model(imgs)
              loss = criterion(preds, masks)
              iou, dice = self.calculate_iou_dice(preds, masks)
              if not math.isnan(loss.item()):
                  val_loss += loss.item()
                  val_iou += iou
                  val_dice += dice
                  valid_batches += 1

      if valid_batches == 0:
          avg_val_iou = 0.0
          avg_val_dice = 0.0
      else:
          avg_val_iou = val_iou / valid_batches
          avg_val_dice = val_dice / valid_batches

      f1_score = 2 * avg_val_iou * avg_val_dice / (avg_val_iou + avg_val_dice + 1e-6)
      print("Значение целевой метрики на валидационной выборке: ", abs(metric))
      print("Значение целевой метрики на тестовой выборке: ", f1_score)

      return f1_score

if __name__ == '__main__':
    problem_class = AirObjectDetectionProblem(3)
    problem = GlobalizerProblem(problem_class)
    result = problem.calculate([-3.0], ["16", "4"])
    problem.finalize([-3.0], ["16", "4"])
    print(result)