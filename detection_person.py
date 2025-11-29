from ultralytics import YOLO
import cv2
import os
from traitement_images import eclaircissement_image


def boxes_intersect(boxA, boxB):
    """Renvoie True si deux boxes se chevauchent"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    return inter_width * inter_height > 0  # True s’il y a recouvrement


def analyse_image(image_path: str) -> bool:
    # Charger le modèle
    model = YOLO("yolov8n.pt")

    # Charger ton image (ou une frame vidéo)
    results = model(eclaircissement_image(image_path))[0]

    # Extraire les détections
    persons = []
    beds = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "person":
            persons.append((x1, y1, x2, y2))
        elif label in ["bed", "beds", "couch"]:
            beds.append((x1, y1, x2, y2))

    # Vérifier si une personne est sur un canapé
    person_on_bed = False
    for bed in beds:
        for person in persons:
            if boxes_intersect(bed, person):
                person_on_bed = True
                break
    print(f"{beds = }\n {persons = }")
    return person_on_bed


def lecture_dossier(dir_path: str):
    images = [f for f in os.listdir(dir_path) if f.endswith((".jpg", ".png"))]
    detection_chute = {}
    for image in images:
        person_in_bed = analyse_image(f"MEDIA/IMG/{image}")
        detect = "no chute" if person_in_bed else f"\033[91mchute\033[0m"
        detection_chute[image] = detection_chute.get(image, detect)
    return detection_chute


print(lecture_dossier("Media/IMG"))
