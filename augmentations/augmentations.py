import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import shutil

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from tqdm import tqdm


# ============================================================
# Env loader (Your code - no changes needed)
# ============================================================
def load_env_file(env_path: Path) -> Dict[str, str]:
    config = {}
    if not env_path.exists():
        raise FileNotFoundError(f"Env file not found: {env_path}")
    with env_path.open("r", encoding="utf-8") as f:
        buf = ""
        for raw in f:
            line = raw.rstrip("\n")
            if line.strip().endswith("\\"):
                buf += line.strip()[:-1]
                continue
            else:
                buf += line
            s = buf.strip()
            buf = ""
            if not s or s.startswith("#"): continue
            if "=" not in s: continue
            key, value = s.split("=", 1)
            config[key.strip()] = value.strip()
    return config


def str_to_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None: return default
    return val.strip().lower() in {"1", "true", "yes", "y"}


def str_to_float(val: Optional[str], default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def str_to_int(val: Optional[str], default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def parse_rgb_triplet(s: str) -> Optional[Tuple[int, int, int]]:
    try:
        r, g, b = [int(x) for x in s.split(",")]
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    except Exception:
        return None


def parse_color_list(val: str) -> List[Tuple[int, int, int]]:
    if not val: return []
    colors: List[Tuple[int, int, int]] = []
    for item in val.split(";"):
        item = item.strip()
        if not item: continue
        c = parse_rgb_triplet(item)
        if c is not None: colors.append(c)
    return colors


def parse_color_combinations(val: str) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    if not val: return []
    combos = []
    for part in val.split(";"):
        part = part.strip()
        if not part or "|" not in part: continue
        a, b = part.split("|", 1)
        ta, tb = parse_rgb_triplet(a.strip()), parse_rgb_triplet(b.strip())
        if ta is not None and tb is not None: combos.append((ta, tb))
    return combos


# ============================================================
# Load config
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent
ENV_PATH = CURRENT_DIR / ".augmentations_env"
CFG = load_env_file(ENV_PATH)

# Output controls
N_AUG_PER_IMAGE = str_to_int(CFG.get("N_AUG_PER_IMAGE"), 2)
KEEP_ORIGINAL = str_to_bool(CFG.get("KEEP_ORIGINAL"), False)
seed_val = (CFG.get("RANDOM_SEED") or "").strip()
if seed_val:
    try:
        s = int(seed_val)
        random.seed(s)
        np.random.seed(s)
    except ValueError:
        pass

# Paths
DATA_DIR = (CURRENT_DIR / CFG.get("DATA_DIR", "../data")).resolve()
AUG_DATA_DIR = (CURRENT_DIR / CFG.get("AUG_DATA_DIR", "../data_augmentations")).resolve()

# Save
SAVE_FORMAT = CFG.get("SAVE_FORMAT", "png").lower()

# Enable/Disable Flags
USE_ROTATION = str_to_bool(CFG.get("USE_ROTATION"), True)
USE_FLIP = str_to_bool(CFG.get("USE_FLIP"), True)
USE_BRIGHTNESS_CONTRAST = str_to_bool(CFG.get("USE_BRIGHTNESS_CONTRAST"), True)
USE_BLUR = str_to_bool(CFG.get("USE_BLUR"), True)
USE_NOISE = str_to_bool(CFG.get("USE_NOISE"), True)
USE_AFFINE = str_to_bool(CFG.get("USE_AFFINE"), True)
USE_INVERT = str_to_bool(CFG.get("USE_INVERT"), True)
USE_HIGHLIGHT_TEXT = str_to_bool(CFG.get("USE_HIGHLIGHT_TEXT"), True)
USE_RANDOM_RESIZE = str_to_bool(CFG.get("USE_RANDOM_RESIZE"), True)

# Color Augmentation Flags
USE_BACKGROUND_REPLACEMENT = str_to_bool(CFG.get("USE_BACKGROUND_REPLACEMENT"), True)  # <--- NEW
USE_COLOR_COMBINATIONS = str_to_bool(CFG.get("USE_COLOR_COMBINATIONS"), True)
USE_SPLIT_COLOR_BG = str_to_bool(CFG.get("USE_SPLIT_COLOR_BG"), True)

# Base Params
MAX_ROTATION_ANGLE = str_to_float(CFG.get("MAX_ROTATION_ANGLE"), 5.0)
BRIGHTNESS_MIN = str_to_float(CFG.get("BRIGHTNESS_MIN"), 0.8)
BRIGHTNESS_MAX = str_to_float(CFG.get("BRIGHTNESS_MAX"), 1.2)
CONTRAST_MIN = str_to_float(CFG.get("CONTRAST_MIN"), 0.8)
CONTRAST_MAX = str_to_float(CFG.get("CONTRAST_MAX"), 1.2)
BLUR_MAX_RADIUS = str_to_float(CFG.get("BLUR_MAX_RADIUS"), 1.2)
NOISE_STD_MIN = str_to_float(CFG.get("NOISE_STD_MIN"), 2.0)
NOISE_STD_MAX = str_to_float(CFG.get("NOISE_STD_MAX"), 10.0)
AFFINE_MAX_TRANSLATE_FRACTION = str_to_float(CFG.get("AFFINE_MAX_TRANSLATE_FRACTION"), 0.02)
AFFINE_SCALE_MIN = str_to_float(CFG.get("AFFINE_SCALE_MIN"), 0.95)
AFFINE_SCALE_MAX = str_to_float(CFG.get("AFFINE_SCALE_MAX"), 1.05)
INVERT_PROB = str_to_float(CFG.get("INVERT_PROB"), 0.15)

# Segmentation
TEXT_IS_DARK = str_to_bool(CFG.get("TEXT_IS_DARK"), True)

# Color Augmentation Params
BG_REPLACEMENT_COLORS = parse_color_list(CFG.get("BG_REPLACEMENT_COLORS", ""))  # <--- NEW
BG_REPLACEMENT_PROB = str_to_float(CFG.get("BG_REPLACEMENT_PROB"), 0.8)  # <--- NEW

COLOR_COMBINATIONS = parse_color_combinations(CFG.get("COLOR_COMBINATIONS", ""))
COLOR_COMBO_PROB = str_to_float(CFG.get("COLOR_COMBO_PROB"), 0.5)

SPLIT_BG_ORIENTATION = (CFG.get("SPLIT_BG_ORIENTATION") or "random").strip().lower()
SPLIT_BG_PROB = str_to_float(CFG.get("SPLIT_BG_PROB"), 0.35)

# Highlight Params
HIGHLIGHT_STROKE_WIDTH_MIN = str_to_int(CFG.get("HIGHLIGHT_STROKE_WIDTH_MIN"), 1)
HIGHLIGHT_STROKE_WIDTH_MAX = str_to_int(CFG.get("HIGHLIGHT_STROKE_WIDTH_MAX"), 3)
HIGHLIGHT_STROKE_COLORS = parse_color_list(CFG.get("HIGHLIGHT_STROKE_COLORS", "255,255,0;0,0,0;255,255,255"))
HIGHLIGHT_PROB = str_to_float(CFG.get("HIGHLIGHT_PROB"), 0.6)
HIGHLIGHT_TEXT_CONTRAST_MIN = str_to_float(CFG.get("HIGHLIGHT_TEXT_CONTRAST_MIN"), 1.0)
HIGHLIGHT_TEXT_CONTRAST_MAX = str_to_float(CFG.get("HIGHLIGHT_TEXT_CONTRAST_MAX"), 1.4)
HIGHLIGHT_TEXT_SHARPNESS_MIN = str_to_float(CFG.get("HIGHLIGHT_TEXT_SHARPNESS_MIN"), 1.0)
HIGHLIGHT_TEXT_SHARPNESS_MAX = str_to_float(CFG.get("HIGHLIGHT_TEXT_SHARPNESS_MAX"), 1.6)

# Random Resize Params
RANDOM_RESIZE_SCALE_MIN = str_to_float(CFG.get("RANDOM_RESIZE_SCALE_MIN"), 0.85)
RANDOM_RESIZE_SCALE_MAX = str_to_float(CFG.get("RANDOM_RESIZE_SCALE_MAX"), 1.20)
RANDOM_RESIZE_RETURN_TO_ORIGINAL = str_to_bool(CFG.get("RANDOM_RESIZE_RETURN_TO_ORIGINAL"), True)


# ============================================================
# Helpers
# ============================================================
def ensure_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def get_text_bg_masks(img_rgb: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    gray = img_rgb.convert("L")
    g = np.array(gray).astype(np.float32)
    thresh = (float(g.min()) + float(g.max())) / 2.0
    text_mask = g < thresh if TEXT_IS_DARK else g > thresh
    bg_mask = ~text_mask
    return text_mask, bg_mask


# ============================================================
# Augmentation Functions
# ============================================================

# <--- NEW: Your primary requested feature ---
def replace_background_color(img: Image.Image) -> Image.Image:
    """
    جایگزینی کامل پس‌زمینه با یک رنگ تصادفی از لیست BG_REPLACEMENT_COLORS.
    متن دست‌نخورده باقی می‌ماند.
    """
    img = ensure_rgb(img)
    _, bg_mask = get_text_bg_masks(img)
    np_img = np.array(img).astype(np.uint8)

    new_bg_color = random.choice(BG_REPLACEMENT_COLORS)
    np_img[bg_mask] = np.array(new_bg_color, dtype=np.uint8)

    return Image.fromarray(np_img)


def apply_color_combination(img: Image.Image) -> Image.Image:
    img = ensure_rgb(img)
    text_mask, bg_mask = get_text_bg_masks(img)
    np_img = np.array(img).astype(np.uint8)
    combo = random.choice(COLOR_COMBINATIONS)
    text_color, bg_color = combo
    np_img[text_mask] = np.array(text_color, dtype=np.uint8)
    np_img[bg_mask] = np.array(bg_color, dtype=np.uint8)
    return Image.fromarray(np_img)


def split_color_background(img: Image.Image) -> Image.Image:
    img = ensure_rgb(img)
    _, bg_mask = get_text_bg_masks(img)
    np_img = np.array(img).astype(np.uint8)

    # برای پس‌زمینه نیم-نیم، از لیست ترکیب‌های رنگی دو رنگ انتخاب می‌کنیم
    combo_a = random.choice(COLOR_COMBINATIONS)
    combo_b = random.choice(COLOR_COMBINATIONS)
    _, bg1 = combo_a
    _, bg2 = combo_b

    h, w = np_img.shape[:2]
    orientation = SPLIT_BG_ORIENTATION
    if orientation == "random": orientation = random.choice(["vertical", "horizontal"])

    if orientation == "vertical":
        mid = w // 2
        left = np.zeros((h, w), dtype=bool);
        left[:, :mid] = True
        mask1 = bg_mask & left;
        mask2 = bg_mask & ~left
    else:  # horizontal
        mid = h // 2
        top = np.zeros((h, w), dtype=bool);
        top[:mid, :] = True
        mask1 = bg_mask & top;
        mask2 = bg_mask & ~top

    np_img[mask1] = np.array(bg1, dtype=np.uint8)
    np_img[mask2] = np.array(bg2, dtype=np.uint8)
    return Image.fromarray(np_img)


# --- Other augmentations from your file (unchanged) ---

def highlight_text(img: Image.Image) -> Image.Image:
    if random.random() > HIGHLIGHT_PROB: return img
    img = ensure_rgb(img)
    text_mask, _ = get_text_bg_masks(img)
    np_img = np.array(img).astype(np.uint8)
    w = random.randint(max(1, HIGHLIGHT_STROKE_WIDTH_MIN), max(1, HIGHLIGHT_STROKE_WIDTH_MAX))
    mask_pil = Image.fromarray((text_mask.astype(np.uint8) * 255), mode="L")
    dilated = mask_pil.filter(ImageFilter.MaxFilter(size=2 * w + 1))
    outline_mask = (np.array(dilated) > 0) & (~text_mask)
    stroke_color = random.choice(HIGHLIGHT_STROKE_COLORS) if HIGHLIGHT_STROKE_COLORS else (255, 255, 0)
    np_img[outline_mask] = np.array(stroke_color, dtype=np.uint8)
    out = Image.fromarray(np_img)
    contrast = random.uniform(HIGHLIGHT_TEXT_CONTRAST_MIN, HIGHLIGHT_TEXT_CONTRAST_MAX)
    sharp = random.uniform(HIGHLIGHT_TEXT_SHARPNESS_MIN, HIGHLIGHT_TEXT_SHARPNESS_MAX)
    enhanced = ImageEnhance.Sharpness(ImageEnhance.Contrast(out).enhance(contrast)).enhance(sharp)
    out_np, enh_np = np.array(out), np.array(enhanced)
    region = text_mask | outline_mask
    out_np[region] = enh_np[region]
    return Image.fromarray(out_np)


def random_resize(img: Image.Image) -> Image.Image:
    orig_w, orig_h = img.size
    scale = random.uniform(RANDOM_RESIZE_SCALE_MIN, RANDOM_RESIZE_SCALE_MAX)
    new_w, new_h = max(8, int(orig_w * scale)), max(8, int(orig_h * scale))
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    if RANDOM_RESIZE_RETURN_TO_ORIGINAL:
        resized = resized.resize((orig_w, orig_h), Image.BICUBIC)
    return resized


def random_rotation(img: Image.Image) -> Image.Image:
    angle = random.uniform(-MAX_ROTATION_ANGLE, MAX_ROTATION_ANGLE)
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def random_flip(img: Image.Image) -> Image.Image:
    if random.random() < 0.5: img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def random_brightness_contrast(img: Image.Image) -> Image.Image:
    b = random.uniform(BRIGHTNESS_MIN, BRIGHTNESS_MAX)
    c = random.uniform(CONTRAST_MIN, CONTRAST_MAX)
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    return img


def random_blur(img: Image.Image) -> Image.Image:
    radius = random.uniform(0, BLUR_MAX_RADIUS)
    return img if radius <= 0.05 else img.filter(ImageFilter.GaussianBlur(radius))


def add_gaussian_noise(img: Image.Image) -> Image.Image:
    img = ensure_rgb(img)
    std = random.uniform(NOISE_STD_MIN, NOISE_STD_MAX)
    np_img = np.array(img).astype(np.float32)
    noise = np.random.normal(0.0, std, np_img.shape)
    np_noisy = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_noisy)


def random_affine(img: Image.Image) -> Image.Image:
    img = ensure_rgb(img)
    w, h = img.size
    max_dx = w * AFFINE_MAX_TRANSLATE_FRACTION
    max_dy = h * AFFINE_MAX_TRANSLATE_FRACTION
    dx, dy = random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)
    scale = random.uniform(AFFINE_SCALE_MIN, AFFINE_SCALE_MAX)
    return img.transform((w, h), Image.AFFINE, data=(scale, 0, dx, 0, scale, dy), resample=Image.BICUBIC,
                         fillcolor=(255, 255, 255))


def maybe_invert(img: Image.Image) -> Image.Image:
    if random.random() > INVERT_PROB: return img
    img = ensure_rgb(img)
    return Image.fromarray((255 - np.array(img)).astype(np.uint8))


# ============================================================
# Pipeline (Improved Logic)
# ============================================================
def apply_random_augmentations(img: Image.Image) -> Image.Image:
    """
    ترتیب منطقی اصلاح شده:
    1) تغییرات هندسی
    2) تغییرات رنگی پس‌زمینه (فقط یکی از سه نوع ممکن)
    3) تغییرات نوری/نویز
    4) برجسته‌سازی نهایی
    """
    img = ensure_rgb(img)

    # 1. Geometric Augmentations
    geo_ops = []
    if USE_RANDOM_RESIZE: geo_ops.append(random_resize)
    if USE_ROTATION: geo_ops.append(random_rotation)
    if USE_FLIP: geo_ops.append(random_flip)
    if USE_AFFINE: geo_ops.append(random_affine)
    random.shuffle(geo_ops)
    for op in geo_ops:
        if random.random() < 0.9:  # Apply with high probability
            img = op(img)

    # 2. Color-based Background Augmentations (Mutually Exclusive)
    # <--- REVISED LOGIC: Ensures only one background change is applied per image.
    possible_bg_changes = []
    if USE_BACKGROUND_REPLACEMENT and BG_REPLACEMENT_COLORS and random.random() < BG_REPLACEMENT_PROB:
        possible_bg_changes.append(replace_background_color)
    if USE_COLOR_COMBINATIONS and COLOR_COMBINATIONS and random.random() < COLOR_COMBO_PROB:
        possible_bg_changes.append(apply_color_combination)
    if USE_SPLIT_COLOR_BG and COLOR_COMBINATIONS and random.random() < SPLIT_BG_PROB:
        possible_bg_changes.append(split_color_background)

    if possible_bg_changes:
        chosen_bg_op = random.choice(possible_bg_changes)
        img = chosen_bg_op(img)

    # 3. Photometric Augmentations
    photo_ops = []
    if USE_BRIGHTNESS_CONTRAST: photo_ops.append(random_brightness_contrast)
    if USE_BLUR: photo_ops.append(random_blur)
    if USE_NOISE: photo_ops.append(add_gaussian_noise)
    if USE_INVERT: photo_ops.append(maybe_invert)
    random.shuffle(photo_ops)
    if photo_ops:
        k = random.randint(0, len(photo_ops))
        for op in photo_ops[:k]:
            img = op(img)

    # 4. Final Touch: Highlighting
    if USE_HIGHLIGHT_TEXT:
        img = highlight_text(img)

    return img


# ============================================================
# Dataset processing (Improved for clarity)
# ============================================================
def process_dataset() -> None:
    if not DATA_DIR.exists():
        print(f"Error: DATA_DIR not found at '{DATA_DIR}'")
        return

    if AUG_DATA_DIR.exists():
        shutil.rmtree(AUG_DATA_DIR)
    AUG_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Source directory: {DATA_DIR}")
    print(f"Destination directory: {AUG_DATA_DIR}")
    print(f"Settings: {N_AUG_PER_IMAGE} augmentations/image | Keep original: {KEEP_ORIGINAL}\n")

    all_image_paths = list(DATA_DIR.rglob("*"))
    image_files = [p for p in all_image_paths if p.is_file() and is_image_file(p)]

    if not image_files:
        print("No image files found in the source directory.")
        return

    for img_path in tqdm(image_files, desc="Augmenting Images"):
        relative_path = img_path.relative_to(DATA_DIR)
        out_dir = AUG_DATA_DIR / relative_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(img_path) as img:
                base = img_path.stem

                if KEEP_ORIGINAL:
                    orig_save_path = out_dir / f"{base}.{SAVE_FORMAT}"
                    ensure_rgb(img).save(orig_save_path)

                for i in range(N_AUG_PER_IMAGE):
                    aug = apply_random_augmentations(img)
                    aug = ensure_rgb(aug)
                    aug_save_path = out_dir / f"{base}_aug_{i + 1}.{SAVE_FORMAT}"
                    aug.save(aug_save_path)
        except Exception as e:
            print(f"\n[Warning] Could not process {img_path}: {e}")

    print("\nAugmentation process completed successfully!")


if __name__ == "__main__":
    process_dataset()
