from ultralytics import YOLO
from PIL import Image
import os

# Crée le dossier "IMG" s’il n’existe pas déjà
output_dir = "IMG"
os.makedirs(output_dir, exist_ok=True)

# Charger le modèle
model = YOLO("yolov8n.pt")

# Image d'entrée
image_path = "IMG/lit_avec_pers.jpg"
results = model(image_path)
results[0].show()

# Ouvrir l'image avec PIL
img = Image.open(image_path)

# Parcourir toutes les détections
count = 0
for box in results[0].boxes:
    cls = int(box.cls[0])
    label = model.names[cls]

    if label in ["couch","bed"]:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img.crop((x1, y1, x2, y2))

        #Crée un sous dossier en fonction de l'objet détecté
        obj_dir = os.path.join(output_dir, label)
        os.makedirs(obj_dir, exist_ok=True)

        # Nom unique pour chaque canapé détecté
        filename = os.path.join(obj_dir, f"{label}_{count + 1}.jpg")
        crop.save(filename)
        print(f"✅ {label.capitalize()} {count+1} sauvegardé : {filename}")
        count += 1

if count == 0:
    print("❌ Aucun canapé ni lit détecté.")
else:
    print(f"🎉 {count} objet(s) détecté(s) et sauvegardé(s) dans le dossier {output_dir}/.")
