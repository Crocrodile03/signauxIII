from ultralytics import YOLO
import cv2
import os
from ameliration_image import ameliorer_image
import numpy as np


def calculate_iou(boxA, boxB):
    """Calcule l'Intersection over Union (IoU) entre deux boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    # Aire de chaque box
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # IoU = intersection / union
    union_area = boxA_area + boxB_area - inter_area
    iou = inter_area / union_area

    return iou


def calculate_overlap_ratio(person_box, bed_box):
    """Calcule le ratio de la personne qui se trouve dans le lit"""
    xA = max(person_box[0], bed_box[0])
    yA = max(person_box[1], bed_box[1])
    xB = min(person_box[2], bed_box[2])
    yB = min(person_box[3], bed_box[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    # Aire de la personne
    person_area = (person_box[2] - person_box[0]) * (person_box[3] - person_box[1])

    # Ratio de la personne dans le lit
    overlap_ratio = inter_area / person_area

    return overlap_ratio


def analyse_image(
    image_path: str, iou_threshold: float = 0.4, overlap_threshold: float = 0.6
) -> bool:
    """
    Analyse une image pour détecter si une personne est dans un lit.

    Args:
        image_path: Chemin de l'image
        iou_threshold: Seuil IoU minimum pour considérer une superposition (défaut: 0.1)
        overlap_threshold: Ratio minimum de la personne qui doit être dans le lit (défaut: 0.3 = 30%)

    Returns:
        True si une personne est détectée dans le lit, False sinon
    """
    # Charger le modèle
    model = YOLO("yolov8n.pt")

    # Améliorer l'image et la convertir en RGB pour YOLO
    img_amelioree = ameliorer_image(image_path)

    # S'assurer que l'image a 3 canaux (RGB)
    if len(img_amelioree.shape) == 2:  # Image en niveaux de gris
        import skimage as skim

        img_amelioree = skim.color.gray2rgb(img_amelioree)
    elif img_amelioree.shape[2] == 1:  # Image avec 1 canal
        img_amelioree = np.repeat(img_amelioree, 3, axis=2)

    # Charger ton image (ou une frame vidéo)
    results = model(img_amelioree)[0]

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

    # Vérifier si une personne est dans/sur un lit
    person_on_bed = False

    for bed in beds:
        for person in persons:
            iou = calculate_iou(bed, person)
            overlap_ratio = calculate_overlap_ratio(person, bed)

            # Une personne est considérée dans le lit si:
            # - Il y a une superposition (IoU > seuil)
            # - OU si au moins X% de la personne est dans le lit
            if iou > iou_threshold or overlap_ratio > overlap_threshold:
                person_on_bed = True
                print(
                    f"Personne détectée dans le lit - IoU: {iou:.2f}, Overlap: {overlap_ratio:.2f}"
                )
                break

        if person_on_bed:
            break

    print(f"{beds = }\n{persons = }")
    print(f"Personne dans le lit: {person_on_bed}")

    return person_on_bed


def lecture_dossier(dir_path: str):
    images = [f for f in os.listdir(dir_path) if f.endswith((".jpg", ".png"))]
    detection_chute = {}
    for image in images:
        person_in_bed = analyse_image(f"MEDIA/IMG/{image}")
        detect = "no chute" if person_in_bed else f"\033[91mchute\033[0m"
        detection_chute[image] = detection_chute.get(image, detect)
    return detection_chute
