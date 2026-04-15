# train.py  ──  Start fresh YOLOv26 + continue from your 3-epoch checkpoint

from ultralytics import YOLO

# ── PATHS ────────────────────────────────────────────────────────────────
BASE_MODEL = "yolo26n.pt"                     # Fresh YOLOv26 nano (2026)

CHECKPOINT_PATH = r"C:\Users\vinit\final-project-vscode\runs\train_416_cpu\weights\last.pt"
# ^ your 3-epoch checkpoint (will be loaded after base model)

DATA_YAML = r"C:\Users\vinit\final project\data.yaml"

# ── TRAINING SETTINGS ────────────────────────────────────────────────────
EPOCHS_TOTAL    = 80                       # total target (including the 3 already done)
IMGSZ           = 320                      # 320 is much faster than 416, accuracy drop usually small
BATCH           = 4                        # small batch = faster on CPU (increase to 8 if RAM allows)
DEVICE          = "cpu"
PATIENCE        = 12                       # stop early if no improvement for 12 epochs
SAVE_PERIOD     = 3                        # save checkpoint every 3 epochs (safer)

# ── FAST + REASONABLE ACCURACY SETTINGS ──────────────────────────────────
AUGMENT_LIGHT = {
    'mosaic': 0.6,          # lower than default 1.0 → faster
    'mixup': 0.1,           # almost off
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 5.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
}

# ── LOAD BASE + CHECKPOINT ───────────────────────────────────────────────
print("Loading fresh YOLOv26 nano base model...")
model = YOLO(BASE_MODEL)

print(f"Loading your 3-epoch checkpoint: {CHECKPOINT_PATH}")
model = YOLO(CHECKPOINT_PATH)               # this overwrites base with your learned weights

print("Starting/continuing training from epoch ~3 ...")

# ── TRAIN ────────────────────────────────────────────────────────────────
model.train(
    data          = DATA_YAML,
    resume        = True,                   # continues from checkpoint epoch
    epochs        = EPOCHS_TOTAL,
    imgsz         = IMGSZ,
    batch         = BATCH,
    device        = DEVICE,
    project       = r"C:\Users\vinit\final-project-vscode\runs",
    name          = "y26n_fast_320_resume",
    patience      = PATIENCE,
    save_period   = SAVE_PERIOD,
    plots         = True,
    exist_ok      = True,
    
    # Light augmentations = faster training, still good generalization
    mosaic        = AUGMENT_LIGHT['mosaic'],
    mixup         = AUGMENT_LIGHT['mixup'],
    hsv_h         = AUGMENT_LIGHT['hsv_h'],
    hsv_s         = AUGMENT_LIGHT['hsv_s'],
    hsv_v         = AUGMENT_LIGHT['hsv_v'],
    degrees       = AUGMENT_LIGHT['degrees'],
    translate     = AUGMENT_LIGHT['translate'],
    scale         = AUGMENT_LIGHT['scale'],
    shear         = AUGMENT_LIGHT['shear'],
    perspective   = AUGMENT_LIGHT['perspective'],
    flipud        = AUGMENT_LIGHT['flipud'],
    fliplr        = AUGMENT_LIGHT['fliplr'],
)

print("Training finished or stopped.")