import os
from ultralytics import YOLO
from PIL import Image
from traitement_images import eclaircissement_image

# Crée le dossier "IMG" s’il n’existe pas déjà
OUTPUT_DIR = "MEDIA/OBJ_DETECT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Charger le modèle
model = YOLO("yolov8n.pt")

# Image d'entrée
IMAGE_PATH = input("path image: ")
results = model(IMAGE_PATH)[0]
results.show()

# Ouvrir l'image avec PIL
img = Image.open(IMAGE_PATH)

# Parcourir toutes les détections
COUNT = 0
for box in results.boxes:
    cls = int(box.cls[0])
    label = model.names[cls]

    if label in ["couch", "bed"]:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img.crop((x1, y1, x2, y2))

        # Crée un sous dossier en fonction de l'objet détecté
        obj_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(obj_dir, exist_ok=True)

        # Nom unique pour chaque canapé détecté
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
