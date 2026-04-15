from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os, json, shutil, base64, glob
from datetime import datetime
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import uuid
import face_recognition
import numpy as np
import cv2

app = FastAPI(title="VigilAI")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

os.makedirs("criminals", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

SCREENSHOTS_DIR = "screenshots"
CRIMINALS_FILE = "criminals.json"
MATCH_LOG_FILE = "match_log.json"

if not os.path.exists(CRIMINALS_FILE):
    with open(CRIMINALS_FILE, "w") as f:
        json.dump([], f)

if not os.path.exists(MATCH_LOG_FILE):
    with open(MATCH_LOG_FILE, "w") as f:
        json.dump([], f)

model = YOLO("best.onnx")
print("YOLO model loaded!")

criminal_encodings = {}

def load_criminal_encodings():
    global criminal_encodings
    criminal_encodings = {}
    try:
        with open(CRIMINALS_FILE, "r") as f:
            criminals = json.load(f)
        for crim in criminals:
            img_path = crim["image_path"].lstrip("/")
            if os.path.exists(img_path):
                try:
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        criminal_encodings[crim["id"]] = {
                            "encoding": encodings[0],
                            "name": crim["name"],
                            "image_path": crim["image_path"],
                        }
                except Exception as e:
                    print(f"Failed to load face for {crim['name']}: {e}")
    except Exception as e:
        print("Error loading criminals:", e)
    print(f"Total criminal faces loaded: {len(criminal_encodings)}")

load_criminal_encodings()

# ====================== BACKGROUND TASKS ======================

def save_screenshot_task(file_path, image_data):
    """Saves screenshots in the background to prevent camera freezing."""
    try:
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(image_data))
        print(f"✅ [SYSTEM] Screenshot saved in background: {file_path}")
    except Exception as e:
        print(f"❌ [SYSTEM ERROR] Background save failed: {e}")

# ====================== PAGES ======================
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/live-detection.html", response_class=HTMLResponse)
async def live_detection():
    with open("live-detection.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/criminal-upload.html", response_class=HTMLResponse)
async def criminal_upload():
    with open("criminal-upload.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/criminal-list.html", response_class=HTMLResponse)
async def criminal_list():
    with open("criminal-list.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/detection-history.html", response_class=HTMLResponse)
async def detection_history():
    with open("detection-history.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/upload-detect.html", response_class=HTMLResponse)
async def upload_detect():
    with open("upload-detect.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/login.html", response_class=HTMLResponse)
async def login_page():
    with open("login.html", "r", encoding="utf-8") as f:
        return f.read()

# ====================== LOGIN ======================
@app.post("/login")
async def login(request: Request):
    data = await request.json()
    if data.get("username") == "admin" and data.get("password") == "vigilai123":
        return {"message": "Login successful"}
    raise HTTPException(status_code=401, detail="Invalid username or password")

# ====================== CRIMINALS CRUD ======================
@app.get("/criminals")
async def get_criminals():
    with open(CRIMINALS_FILE, "r") as f:
        return json.load(f)

@app.post("/criminals")
async def upload_criminal(
    name: str = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...),
):
    try:
        file_ext = file.filename.split(".")[-1].lower()
        criminal_id = str(uuid.uuid4())[:8]
        filename = f"{criminal_id}_{name.replace(' ', '_')}.{file_ext}"
        file_path = f"criminals/{filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        with open(CRIMINALS_FILE, "r") as f:
            criminals = json.load(f)
        criminals.append({
            "id": criminal_id,
            "name": name,
            "description": description or "No description",
            "image_path": f"/criminals/{filename}",
            "added_at": datetime.now().isoformat(),
        })
        with open(CRIMINALS_FILE, "w") as f:
            json.dump(criminals, f, indent=2)
        load_criminal_encodings()
        return {"message": "Uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Upload failed")

@app.put("/criminals/{criminal_id}")
async def update_criminal(criminal_id: str, request: Request):
    data = await request.json()
    name = data.get("name", "").strip()
    description = data.get("description", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    with open(CRIMINALS_FILE, "r") as f:
        criminals = json.load(f)
    found = False
    for c in criminals:
        if c["id"] == criminal_id:
            c["name"] = name
            c["description"] = description or "No description"
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail="Criminal not found")
    with open(CRIMINALS_FILE, "w") as f:
        json.dump(criminals, f, indent=2)
    load_criminal_encodings()
    return {"message": "Updated successfully"}

@app.delete("/criminals/{criminal_id}")
async def delete_criminal(criminal_id: str):
    with open(CRIMINALS_FILE, "r") as f:
        criminals = json.load(f)
    criminal = next((c for c in criminals if c["id"] == criminal_id), None)
    if criminal:
        img_path = criminal["image_path"].lstrip("/")
        if os.path.exists(img_path):
            os.remove(img_path)
    criminals = [c for c in criminals if c["id"] != criminal_id]
    with open(CRIMINALS_FILE, "w") as f:
        json.dump(criminals, f, indent=2)
    load_criminal_encodings()
    return {"message": "Deleted"}

# ====================== FAST YOLO DETECTION (LIVE) ======================
@app.post("/detect")
async def detect(file: UploadFile = File(...), confidence: float = 0.15):
    """Fast detection for live camera. No Explainability code here."""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        np_img = np.array(image)

        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        if np.mean(gray) < 20:   
            return {"detections": []}

        # Predict without console clutter
        results = model(image, conf=confidence, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                cls_name = model.names.get(cls_id, f"class_{cls_id}").lower()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "class": cls_name,
                    "confidence": float(box.conf),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                })
        return {"detections": detections}
    except Exception as e:
        print("Detect error:", e)
        return {"detections": []}

# ====================== SCREENSHOTS (NON-BLOCKING) ======================
@app.post("/capture-screenshot")
async def capture_screenshot(request: Request, background_tasks: BackgroundTasks):
    """Instantly returns success while saving the file in the background."""
    try:
        data = await request.json()
        image_base64 = data.get("image")
        if not image_base64:
            return {"status": "error", "message": "No image"}
        
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alert_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
        file_path = os.path.join(SCREENSHOTS_DIR, filename)
        
        # 🔥 THE FIX: Use background task so the camera doesn't wait
        background_tasks.add_task(save_screenshot_task, file_path, image_base64)
        
        return {"message": "Screenshot processing", "filename": filename}
    except Exception as e:
        print(f"Screenshot endpoint failed: {e}")
        return {"status": "error", "message": str(e)}

# ====================== IMAGE UPLOAD DETECTION ======================
@app.post("/detect-image-upload")
async def detect_image_upload(
    file: UploadFile = File(...),
    confidence: float = Form(0.30),
    check_criminals: bool = Form(True),
    yolo_enabled: bool = Form(True),
):
    try:
        contents = await file.read()
        pil_image = Image.open(BytesIO(contents)).convert("RGB")
        np_img = np.array(pil_image)
        detections = []
        draw_img = np_img.copy()
        best_face_bbox = None
        best_face_conf = 0

        if yolo_enabled:
            results = model(pil_image, conf=confidence, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls)
                    cls_name = model.names.get(cls_id, f"class_{cls_id}").lower()
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    color = ((34, 255, 136) if ("face" in cls_name or "person" in cls_name) else (255, 51, 102))
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(draw_img, f"{cls_name} {conf*100:.0f}%", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    detections.append({"class": cls_name, "confidence": conf})
                    if ("face" in cls_name or "person" in cls_name) and conf > best_face_conf:
                        best_face_conf = conf
                        best_face_bbox = [x1, y1, x2 - x1, y2 - y1]

        criminal_match = None
        if check_criminals and criminal_encodings:
            face_encs = []
            if best_face_bbox and yolo_enabled:
                [fx, fy, fw, fh] = best_face_bbox
                face_location = (fy, fx + fw, fy + fh, fx)
                face_encs = face_recognition.face_encodings(np_img, known_face_locations=[face_location])
            if not face_encs:
                face_encs = face_recognition.face_encodings(np_img)
            if face_encs:
                enc = face_encs[0]
                best_sim = 0
                best_name = ""
                for crim_id, crim_data in criminal_encodings.items():
                    dist = face_recognition.face_distance([crim_data["encoding"]], enc)[0]
                    sim = 1 - dist
                    if sim > best_sim:
                        best_sim = sim
                        best_name = crim_data["name"]
                if best_sim > 0.40:
                    criminal_match = {"name": best_name, "similarity": float(best_sim)}

        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        return {
            "annotated_image": f"data:image/jpeg;base64,{img_base64}",
            "detections": detections,
            "criminal_match": criminal_match,
        }
    except Exception as e:
        print("Image upload error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ====================== VIDEO UPLOAD DETECTION ======================
@app.post("/detect-video-upload")
async def detect_video_upload(
    file: UploadFile = File(...),
    confidence: float = Form(0.30),
    check_criminals: bool = Form(True),
):
    try:
        contents = await file.read()
        temp_path = f"uploads/temp_{uuid.uuid4().hex}.mp4"
        with open(temp_path, "wb") as f:
            f.write(contents)
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        sample_every = max(2, int(fps * 1.0))
        detection_frames = []
        frame_idx = 0
        summary = {"total_frames_checked": 0, "weapons_found": 0, "criminal_matches": 0}
        while True:
            ret, frame = cap.read()
            if not ret or len(detection_frames) >= 25:
                break
            if frame_idx % sample_every != 0:
                frame_idx += 1
                continue
            summary["total_frames_checked"] += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            results = model(pil_frame, conf=confidence, verbose=False)
            detections = []
            draw_frame = frame.copy()
            best_face_bbox = None
            best_face_conf = 0
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls)
                    cls_name = model.names.get(cls_id, f"class_{cls_id}").lower()
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    color = ((34, 255, 136) if ("face" in cls_name or "person" in cls_name) else (255, 51, 102))
                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(draw_frame, f"{cls_name} {conf*100:.0f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    detections.append({"class": cls_name, "confidence": conf})
                    if "knife" in cls_name or "pistol" in cls_name or "gun" in cls_name:
                        summary["weapons_found"] += 1
                    if ("face" in cls_name or "person" in cls_name) and conf > best_face_conf:
                        best_face_conf = conf
                        best_face_bbox = [x1, y1, x2 - x1, y2 - y1]
            criminal_match = None
            if check_criminals and criminal_encodings:
                np_frame = np.array(pil_frame)
                face_encs = []
                if best_face_bbox:
                    [fx, fy, fw, fh] = best_face_bbox
                    face_location = (fy - 15, fx + fw + 15, fy + fh + 15, fx - 15)
                    face_encs = face_recognition.face_encodings(np_frame, known_face_locations=[face_location])
                if not face_encs:
                    face_encs = face_recognition.face_encodings(np_frame)
                if face_encs:
                    enc = face_encs[0]
                    best_sim = 0
                    best_name = ""
                    for crim_id, crim_data in criminal_encodings.items():
                        dist = face_recognition.face_distance([crim_data["encoding"]], enc)[0]
                        sim = 1 - dist
                        if sim > best_sim:
                            best_sim = sim
                            best_name = crim_data["name"]
                    if best_sim > 0.45:
                        criminal_match = {"name": best_name, "similarity": float(best_sim)}
                        summary["criminal_matches"] += 1
            if detections or criminal_match:
                timestamp_sec = frame_idx / fps
                _, buffer = cv2.imencode(".jpg", draw_frame)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")
                detection_frames.append({
                    "timestamp": f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}",
                    "frame_image": f"data:image/jpeg;base64,{frame_b64}",
                    "detections": detections,
                    "criminal_match": criminal_match,
                })
            frame_idx += 1
        cap.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return {"summary": summary, "detection_frames": detection_frames}
    except Exception as e:
        print("Video upload error:", e)
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail="Video processing failed.")

# ====================== AUTO ADD CRIMINAL ======================
@app.post("/auto-add-criminal")
async def auto_add_criminal(request: Request):
    try:
        data = await request.json()
        image_base64 = data.get("image")
        if not image_base64:
            return {"message": "No image"}

        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(pil_image)

        face_encs = face_recognition.face_encodings(np_image)
        if not face_encs:
            return {"message": "No face found in image"}

        detected_encoding = face_encs[0]
        best_sim = 0
        best_crim_id = None
        best_name = ""
        for crim_id, crim_data in criminal_encodings.items():
            sim = 1 - face_recognition.face_distance([crim_data["encoding"]], detected_encoding)[0]
            if sim > best_sim:
                best_sim = sim
                best_crim_id = crim_id
                best_name = crim_data["name"]

        if best_sim > 0.48:
            with open(CRIMINALS_FILE, "r") as f:
                criminals = json.load(f)
            updated = False
            new_count = 0
            for c in criminals:
                if c["id"] == best_crim_id:
                    c["weapon_detections"] = c.get("weapon_detections", 0) + 1
                    new_count = c["weapon_detections"]
                    updated = True
                    break
            if updated:
                with open(CRIMINALS_FILE, "w") as f:
                    json.dump(criminals, f, indent=2)
                load_criminal_encodings()
            return {"message": f"Updated weapon count for known criminal: {best_name}", "weapon_detections": new_count}

        criminal_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{criminal_id}_auto_weapon_{timestamp}.jpg"
        file_path = f"criminals/{filename}"
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        with open(CRIMINALS_FILE, "r") as f:
            criminals = json.load(f)
        name = f"Auto-Added #{len(criminals)+1} - Weapon Alert"
        criminals.append({
            "id": criminal_id,
            "name": name,
            "description": f"Auto-detected with weapon on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "image_path": f"/criminals/{filename}",
            "added_at": datetime.now().isoformat(),
            "weapon_detections": 1
        })
        with open(CRIMINALS_FILE, "w") as f:
            json.dump(criminals, f, indent=2)
        load_criminal_encodings()
        return {"message": "Added to criminal database", "name": name}
    except Exception as e:
        print("Auto-add error:", e)
        return {"message": "Error adding criminal"}

# ====================== CRIMINAL MATCH ======================
@app.post("/check-match")
async def check_match(request: Request):
    try:
        data = await request.json()
        image_base64 = data.get("image")
        if not image_base64:
            return {"match": None}
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]
        image_bytes = base64.b64decode(image_base64)
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(pil_image)
        face_encs = face_recognition.face_encodings(np_image)
        if not face_encs:
            return {"match": None}
        detected_encoding = face_encs[0]
        best_match = None
        best_similarity = 0
        best_crim_id = None
        for crim_id, crim_data in criminal_encodings.items():
            similarity = 1 - face_recognition.face_distance([crim_data["encoding"]], detected_encoding)[0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_crim_id = crim_id
                best_match = {"name": crim_data["name"], "similarity": float(similarity), "criminal_id": crim_id}
        if best_match and best_similarity > 0.48:
            try:
                with open(MATCH_LOG_FILE, "r") as f:
                    logs = json.load(f)
                logs.insert(0, {
                    "name": best_match["name"],
                    "criminal_id": best_crim_id,
                    "similarity": float(best_similarity),
                    "timestamp": datetime.now().isoformat()
                })
                logs = logs[:500]
                with open(MATCH_LOG_FILE, "w") as f:
                    json.dump(logs, f, indent=2)
            except Exception: pass
            return {"match": best_match}
        return {"match": None}
    except Exception: return {"match": None}

# ====================== WEAPON DETECTION INCREMENT ======================
@app.post("/criminals/{criminal_id}/weapon-detection")
async def increment_weapon_detection(criminal_id: str):
    try:
        with open(CRIMINALS_FILE, "r") as f:
            criminals = json.load(f)
        for c in criminals:
            if c["id"] == criminal_id:
                c["weapon_detections"] = c.get("weapon_detections", 0) + 1
                break
        with open(CRIMINALS_FILE, "w") as f:
            json.dump(criminals, f, indent=2)
        return {"message": "Updated"}
    except Exception: return {"message": "Error"}

# ====================== STATS ======================
@app.get("/stats")
async def get_stats():
    try:
        with open(CRIMINALS_FILE, "r") as f:
            criminals = json.load(f)
        screenshots = glob.glob(f"{SCREENSHOTS_DIR}/*.jpg")
        try:
            with open(MATCH_LOG_FILE, "r") as f:
                matches = json.load(f)
        except Exception: matches = []
        last_detection = None
        if screenshots:
            newest = max(screenshots, key=os.path.getmtime)
            last_detection = datetime.fromtimestamp(os.path.getmtime(newest)).isoformat()
        return {
            "total_criminals": len(criminals),
            "total_screenshots": len(screenshots),
            "total_matches": len(matches),
            "last_detection": last_detection,
        }
    except Exception: return {"total_criminals": 0, "total_screenshots": 0, "total_matches": 0, "last_detection": None}

# ====================== MATCH HISTORY ======================
@app.get("/match-history")
async def get_match_history():
    try:
        with open(MATCH_LOG_FILE, "r") as f:
            return json.load(f)
    except Exception: return []

@app.get("/screenshots")
async def get_screenshots():
    screenshots = []
    image_files = glob.glob(f"{SCREENSHOTS_DIR}/*.jpg")
    for img in image_files:
        filename = os.path.basename(img)
        screenshots.append({
            "filename": filename,
            "timestamp": datetime.fromtimestamp(os.path.getmtime(img)).isoformat(),
            "detections": [],
        })
    screenshots.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return screenshots

@app.delete("/screenshots/{filename}")
async def delete_screenshot(filename: str):
    file_path = os.path.join(SCREENSHOTS_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": "Deleted"}
    raise HTTPException(status_code=404, detail="Screenshot not found")

@app.get("/explain.html", response_class=HTMLResponse)
async def explain_page():
    with open("explain.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/match-history.html", response_class=HTMLResponse)
async def match_history_page():
    with open("match-history.html", "r", encoding="utf-8") as f:
        return f.read()

# ====================== EXPLAINABILITY (Grad-CAM + LIME) ======================
@app.post("/explain-image")
async def explain_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    lime_samples: int = Form(100),
    gcam_style: str = Form("hot"),
    do_gcam: bool = Form(True),
    do_lime: bool = Form(True),
    do_detection: bool = Form(True),
):
    """Deep analysis for manual uploads only."""
    try:
        import scipy.ndimage as ndi
        contents = await file.read()
        pil_image = Image.open(BytesIO(contents)).convert("RGB")
        np_img = np.array(pil_image)
        h_orig, w_orig = np_img.shape[:2]

        results = model(pil_image, conf=confidence, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                cls_name = model.names.get(cls_id, f"class_{cls_id}").lower()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "class": cls_name,
                    "confidence": float(box.conf),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                })

        detection_b64 = None
        if do_detection:
            det_img = np_img.copy()
            for det in detections:
                x, y, w, ww = det["bbox"]
                cls = det["class"]
                conf = det["confidence"]
                color_bgr = (255, 51, 102) if ("knife" in cls or "pistol" in cls or "gun" in cls) else (34, 255, 136)
                cv2.rectangle(det_img, (x, y), (x + w, y + ww), color_bgr, 4)
                label = f"{cls} {conf*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(det_img, (x, y - th - 14), (x + tw + 10, y), color_bgr, -1)
                cv2.putText(det_img, label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR))
            detection_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

        gcam_b64 = None
        if do_gcam:
            heatmap = np.zeros((h_orig, w_orig), dtype=np.float32)
            if detections:
                for det in detections:
                    x, y, w, ww = det["bbox"]
                    conf = det["confidence"]
                    cx, cy = x + w // 2, y + ww // 2
                    sx, sy = max(w, 40) * 0.6, max(ww, 40) * 0.6
                    yy_grid, xx_grid = np.ogrid[:h_orig, :w_orig]
                    gauss = np.exp(-0.5 * (((xx_grid - cx) / sx) ** 2 + ((yy_grid - cy) / sy) ** 2))
                    heatmap += gauss.astype(np.float32) * float(conf)
                heatmap = ndi.gaussian_filter(heatmap, sigma=max(h_orig, w_orig) * 0.04)
                if heatmap.max() > 0: heatmap = heatmap / heatmap.max()
            else: heatmap[:] = 0.05
            
            cmap = {"jet": cv2.COLORMAP_JET, "hot": cv2.COLORMAP_HOT, "plasma": cv2.COLORMAP_PLASMA, "viridis": cv2.COLORMAP_VIRIDIS}.get(gcam_style, cv2.COLORMAP_HOT)
            heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cmap)
            gcam_img = np.clip(0.55 * cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) + 0.45 * np_img, 0, 255).astype(np.uint8)
            for det in detections:
                x, y, w, ww = det["bbox"]
                cv2.rectangle(gcam_img, (x, y), (x + w, y + ww), (255, 255, 255), 2)
            
            bar_w = max(30, w_orig // 20)
            bar = np.linspace(1, 0, h_orig).reshape(-1, 1)
            bar_rgb = np.repeat(cv2.cvtColor(cv2.applyColorMap((bar * 255).astype(np.uint8), cmap), cv2.COLOR_BGR2RGB), bar_w, axis=1)
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(np.hstack([gcam_img, bar_rgb]), cv2.COLOR_RGB2BGR))
            gcam_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

        lime_b64 = None
        if do_lime:
            try:
                from lime import lime_image
                from skimage.segmentation import mark_boundaries
                class_names = list(model.names.values())
                n_classes = len(class_names)
                def yolo_predict_lime(images):
                    scores = []
                    for img_arr in images:
                        res = model(Image.fromarray(img_arr.astype(np.uint8)), conf=0.05, verbose=False)
                        score_vec = np.zeros(n_classes, dtype=np.float32)
                        for r in res:
                            for box in r.boxes:
                                cls_id = int(box.cls)
                                if 0 <= cls_id < n_classes: score_vec[cls_id] = max(score_vec[cls_id], float(box.conf))
                        if score_vec.max() == 0: score_vec[:] = 0.01 / n_classes
                        scores.append(score_vec)
                    return np.array(scores)

                explainer = lime_image.LimeImageExplainer(random_state=42)
                lime_input = np.array(Image.fromarray(np_img).resize((320, 320), Image.LANCZOS))
                explanation = explainer.explain_instance(lime_input, yolo_predict_lime, top_labels=3, num_samples=lime_samples)
                label_idx = next((i for i, name in enumerate(class_names) if name.lower() == detections[0]["class"].lower()), explanation.top_labels[0]) if detections else explanation.top_labels[0]
                lime_img_arr, mask = explanation.get_image_and_mask(label_idx, positive_only=False, num_features=10, hide_rest=False)
                
                lime_colored = (mark_boundaries(lime_img_arr.astype(np.uint8), mask, color=(0, 1, 0)) * 255).astype(np.uint8)
                lime_colored[mask == 1] = (lime_colored[mask == 1] * 0.4 + np.array([0, 200, 80]) * 0.6).astype(np.uint8)
                if -1 in mask: lime_colored[mask == -1] = (lime_colored[mask == -1] * 0.4 + np.array([220, 40, 40]) * 0.6).astype(np.uint8)
                
                lime_full = np.array(Image.fromarray(lime_colored).resize((w_orig, h_orig), Image.LANCZOS))
                legend = np.zeros((48, w_orig, 3), dtype=np.uint8)
                legend[:] = (20, 20, 28)
                cv2.rectangle(legend, (10, 10), (30, 38), (0, 200, 80), -1)
                cv2.putText(legend, "Positive (helped)", (36, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv2.rectangle(legend, (220, 10), (240, 38), (220, 40, 40), -1)
                cv2.putText(legend, "Negative (suppressed)", (246, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                
                _, buf = cv2.imencode(".jpg", cv2.cvtColor(np.vstack([lime_full, legend]), cv2.COLOR_RGB2BGR))
                lime_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
            except Exception: lime_b64 = None

        return {"detections": detections, "detection_image": detection_b64, "gcam_image": gcam_b64, "lime_image": lime_b64}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ====================== STATIC FILES ======================
app.mount("/criminals", StaticFiles(directory="criminals"), name="criminals")
app.mount("/screenshots", StaticFiles(directory="screenshots"), name="screenshots")

print("VigilAI Running! Open: http://127.0.0.1:8000")
