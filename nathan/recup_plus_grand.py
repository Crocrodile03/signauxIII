from ultralytics import YOLO
from PIL import Image
import os
from fonction import nathan_trouve_nom_libre

model = YOLO("yolov8n.pt")

image_path = "IMG/lit_avec_pers.jpg"
results = model(image_path)
results[0].show()

output_dir = "IMG"

img = Image.open(image_path)
largest_objects = {}

for box in results[0].boxes:
    cls = int(box.cls[0])
    label = model.names[cls]

    if label in ["couch", "bed"]:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1
        surface = width * height

        if label not in largest_objects or surface > largest_objects[label][0]:
            largest_objects[label] = (surface, (x1, y1, x2, y2))

for label, (_, coords) in largest_objects.items():
    x1, y1, x2, y2 = coords
    crop = img.crop((x1, y1, x2, y2))

    obj_dir = os.path.join(output_dir, label)

    filename = nathan_trouve_nom_libre(obj_dir, label)
    crop.save(filename)
    print(f"✅ Plus grand {label} sauvegardé : {filename}")

print("🎉 Traitement terminé.")
