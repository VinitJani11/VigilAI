# final_precision_boost.py

from ultralytics import YOLO

START_MODEL = r"C:\Users\vinit\final-project-vscode\runs\y26n_precision_boost\weights\last.pt"
DATA_YAML = r"C:\Users\vinit\final project\data.yaml"

model = YOLO(START_MODEL)

model.train(
    data          = DATA_YAML,
    epochs        = 10,                    # short run
    imgsz         = 416,                   # higher resolution for better boxes
    batch         = 4,
    device        = "cpu",
    project       = r"C:\Users\vinit\final-project-vscode\runs",
    name          = "y26n_precision_boost",
    patience      = 5,
    save_period   = 1,
    plots         = True,
    exist_ok      = True,

    # Settings for better box precision
    mosaic        = 0.9,
    mixup         = 0.1,
    degrees       = 5.0,
    translate     = 0.1,
    scale         = 0.6,
    shear         = 1.0,
    perspective   = 0.0,
    flipud        = 0.0,
    fliplr        = 0.5,
)