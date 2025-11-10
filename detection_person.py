from ultralytics import YOLO
import cv2

# Charger le modèle
model = YOLO("yolov8n.pt")

# Charger ton image (ou une frame vidéo)
image_path = "IMG/lit_avec_pers.jpg"
results = model(image_path)[0]

# Extraire les détections
persons = []
couches = []

for box in results.boxes:
    cls = int(box.cls[0])
    label = model.names[cls]

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    if label == "person":
        persons.append((x1, y1, x2, y2))
    elif label == "couch":
        couches.append((x1, y1, x2, y2))

def boxes_intersect(boxA, boxB):
    """Renvoie True si deux boxes se chevauchent"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    return inter_width * inter_height > 0  # True s’il y a recouvrement

# Vérifier si une personne est sur un canapé
person_on_couch = False
for couch in couches:
    for person in persons:
        if boxes_intersect(couch, person):
            person_on_couch = True
            break

if person_on_couch:
    print("Une personne est sur le canapé")
else:
    print("Le canapé est vide")