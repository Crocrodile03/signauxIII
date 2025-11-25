import os
from ultralytics import YOLO
from PIL import Image

OUTPUT_DIR = "VID/IMG"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")

IMAGE_PATH = "MEDIA/IMG/lit_avec_pers.jpg"
results = model(IMAGE_PATH)
results[0].show()

img = Image.open(IMAGE_PATH)

COUNT = 0
for box in results[0].boxes:
    cls = int(box.cls[0])
    label = model.names[cls]

    if label in ["couch", "bed"]:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img.crop((x1, y1, x2, y2))

        obj_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(obj_dir, exist_ok=True)

        filename = os.path.join(obj_dir, f"{label}_{COUNT + 1}.jpg")
        crop.save(filename)
        print(f"✅ {label.capitalize()} {COUNT+1} sauvegardé : {filename}")
        COUNT += 1

if COUNT == 0:
    print("❌ Aucun canapé ni lit détecté.")
else:
    print(
        f"🎉 {COUNT} objet(s) détecté(s) et sauvegardé(s) dans le dossier {OUTPUT_DIR}/."
    )
