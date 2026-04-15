import os
from ultralytics import YOLO

# 1. Define the exact path to your trained weights
# Note: I'm using a 'raw string' (r'') to handle the Windows backslashes correctly
model_path = r'C:\Users\vinit\final-project-vscode\final code\runs\train_416_cpu\weights\best.pt'

# 2. Check if the file exists before starting
if not os.path.exists(model_path):
    print(f"Error: Could not find the file at {model_path}")
else:
    # 3. Load the model
    model = YOLO(model_path)

    # 4. Export to ONNX
    # imgsz=416 matches your folder name 'train_416', which is good for speed!
    print("Starting conversion...")
    success = model.export(
        format='onnx', 
        imgsz=416, 
        simplify=True, 
        opset=12
    )

    if success:
        print(f"Success! Your file is now saved as: {model_path.replace('.pt', '.onnx')}")