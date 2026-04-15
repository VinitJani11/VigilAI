# final_knife_boost.py  ── Short 10-epoch boost for knife improvement

from ultralytics import YOLO

# ── PATHS ────────────────────────────────────────────────────────────────
START_MODEL = r"C:\Users\vinit\final-project-vscode\runs\train_416_cpu\weights\best.pt"

DATA_YAML = r"C:\Users\vinit\final project\data.yaml"

# ── SETTINGS ─────────────────────────────────────────────────────────────
EXTRA_EPOCHS   = 10
IMGSZ          = 320
BATCH          = 4
DEVICE         = "cpu"
PATIENCE       = 5
SAVE_PERIOD    = 2

# Strong knife-focused augmentations
AUGMENT_KNIFE = {
    'mosaic': 1.0,
    'mixup': 0.2,
    'hsv_h': 0.03,
    'hsv_s': 0.95,
    'hsv_v': 0.7,
    'degrees': 10.0,
    'translate': 0.25,
    'scale': 0.8,
    'shear': 3.0,
    'perspective': 0.0015,
    'flipud': 0.0,
    'fliplr': 0.5,
}

# ── LOAD & START NEW SHORT RUN ───────────────────────────────────────────
print("Loading best model from 60 epochs...")
model = YOLO(START_MODEL)

print(f"Starting final 10-epoch knife boost...")

model.train(
    data          = DATA_YAML,
    epochs        = EXTRA_EPOCHS,           # Only 10 new epochs
    imgsz         = IMGSZ,
    batch         = BATCH,
    device        = DEVICE,
    project       = r"C:\Users\vinit\final-project-vscode\runs",
    name          = "y26n_final_knife_boost",
    patience      = PATIENCE,
    save_period   = SAVE_PERIOD,
    plots         = True,
    exist_ok      = True,

    # Knife-focused augmentations
    mosaic        = AUGMENT_KNIFE['mosaic'],
    mixup         = AUGMENT_KNIFE['mixup'],
    hsv_h         = AUGMENT_KNIFE['hsv_h'],
    hsv_s         = AUGMENT_KNIFE['hsv_s'],
    hsv_v         = AUGMENT_KNIFE['hsv_v'],
    degrees       = AUGMENT_KNIFE['degrees'],
    translate     = AUGMENT_KNIFE['translate'],
    scale         = AUGMENT_KNIFE['scale'],
    shear         = AUGMENT_KNIFE['shear'],
    perspective   = AUGMENT_KNIFE['perspective'],
    flipud        = AUGMENT_KNIFE['flipud'],
    fliplr        = AUGMENT_KNIFE['fliplr'],
)

print("Final knife boost completed.")