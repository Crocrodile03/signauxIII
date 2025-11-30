from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from ameliration_image import ameliorer_image


class DetectionResult:
    """Résultat d'une détection"""

    def __init__(
        self,
        image_path: str,
        person_in_bed: bool,
        persons: List[Tuple],
        beds: List[Tuple],
        iou: float = 0.0,
        overlap: float = 0.0,
        status_message: str = "",
    ):
        self.image_path = image_path
        self.person_in_bed = person_in_bed
        self.persons = persons
        self.beds = beds
        self.iou = iou
        self.overlap = overlap
        self.status_message = status_message

    def __repr__(self):
        if self.status_message:
            return f"{Path(self.image_path).name}: ⚠️ {self.status_message}"
        status = "✅ Dans le lit" if self.person_in_bed else "🚨 CHUTE"
        return f"{Path(self.image_path).name}: {status} (IoU={self.iou:.2f}, Overlap={self.overlap:.2f})"


def calculate_iou(boxA: Tuple, boxB: Tuple) -> float:
    """Calcule l'Intersection over Union (IoU) entre deux boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def calculate_overlap_ratio(person_box: Tuple, bed_box: Tuple) -> float:
    """Calcule le ratio de la personne qui se trouve dans le lit"""
    xA = max(person_box[0], bed_box[0])
    yA = max(person_box[1], bed_box[1])
    xB = min(person_box[2], bed_box[2])
    yB = min(person_box[3], bed_box[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    person_area = (person_box[2] - person_box[0]) * (person_box[3] - person_box[1])
    return inter_area / person_area if person_area > 0 else 0.0


def analyse_image(
    image_path: str,
    model: YOLO = None,
    iou_threshold: float = 0.3,
    overlap_threshold: float = 0.5,
    verbose: bool = True,
    use_preprocessing: bool = True,
) -> DetectionResult:
    """
    Analyse une image pour détecter si une personne est dans un lit.

    Args:
        image_path: Chemin de l'image
        model: Modèle YOLO (créé automatiquement si None)
        iou_threshold: Seuil IoU minimum
        overlap_threshold: Ratio minimum de la personne dans le lit
        verbose: Afficher les détails

    Returns:
        DetectionResult avec les informations de détection
    """
    if model is None:
        model = YOLO("yolov8n.pt")

    # Tester AVEC et SANS prétraitement
    if use_preprocessing:
        if verbose:
            print("🔄 Utilisation du prétraitement d'image...")
        img_amelioree = ameliorer_image(image_path, retourner_uint8=True)
    else:
        if verbose:
            print("📷 Utilisation de l'image originale...")
        img_original = cv2.imread(image_path)
        img_amelioree = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

    # Détection avec confiance abaissée
    results = model(img_amelioree, conf=0.25, verbose=False)[
        0
    ]  # conf=0.25 au lieu de 0.5

    persons = []
    beds = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "person":
            persons.append((x1, y1, x2, y2))
        elif label in ["bed", "couch", "suitcase", "bench"]:
            beds.append((x1, y1, x2, y2))

    # Vérifier si les éléments nécessaires sont détectés
    if not persons and not beds:
        if verbose:
            print("⚠️ Aucune personne ni lit détecté")
        return DetectionResult(
            image_path, False, persons, beds, 0.0, 0.0, "Aucune personne ni lit detecte"
        )

    if not persons:
        if verbose:
            print("⚠️ Aucune personne détectée")
        return DetectionResult(
            image_path, False, persons, beds, 0.0, 0.0, "Aucune personne detectee"
        )

    if not beds:
        if verbose:
            print("⚠️ Aucun lit détecté")
        return DetectionResult(
            image_path, False, persons, beds, 0.0, 0.0, "Aucun lit detecte"
        )

    # Vérification personne dans lit (si les deux sont présents)
    person_in_bed = False
    max_iou = 0.0
    max_overlap = 0.0

    for bed in beds:
        for person in persons:
            iou = calculate_iou(bed, person)
            overlap = calculate_overlap_ratio(person, bed)

            max_iou = max(max_iou, iou)
            max_overlap = max(max_overlap, overlap)

            if iou > iou_threshold or overlap > overlap_threshold:
                person_in_bed = True
                if verbose:
                    print(
                        f"✅ Personne dans le lit - IoU: {iou:.2f}, Overlap: {overlap:.2f}"
                    )
                break

        if person_in_bed:
            break

    if verbose and not person_in_bed:
        print(f"🚨 CHUTE DÉTECTÉE - {len(persons)} personne(s), {len(beds)} lit(s)")

    return DetectionResult(
        image_path, person_in_bed, persons, beds, max_iou, max_overlap
    )


def dessiner_detections(
    image_path: str, result: DetectionResult, output_path: str = None
) -> np.ndarray:
    """
    Dessine les détections sur l'image

    Args:
        image_path: Chemin de l'image originale
        result: Résultat de la détection
        output_path: Chemin de sauvegarde (optionnel)

    Returns:
        Image avec les détections dessinées
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Si message de statut (détection incomplète)
    if result.status_message:
        color = (255, 165, 0)  # Orange
        cv2.putText(
            img, result.status_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return img

    # Dessiner les lits en bleu
    for bed in result.beds:
        cv2.rectangle(img, (bed[0], bed[1]), (bed[2], bed[3]), (0, 0, 255), 2)
        cv2.putText(
            img,
            "Lit",
            (bed[0], bed[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # Dessiner les personnes (vert si dans lit, rouge sinon)
    for person in result.persons:
        color = (0, 255, 0) if result.person_in_bed else (255, 0, 0)
        cv2.rectangle(img, (person[0], person[1]), (person[2], person[3]), color, 2)
        label = "Personne (OK)" if result.person_in_bed else "CHUTE!"
        cv2.putText(
            img,
            label,
            (person[0], person[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    # Ajouter le statut global
    status = "Dans le lit" if result.person_in_bed else "CHUTE DETECTEE"
    color = (0, 255, 0) if result.person_in_bed else (255, 0, 0)
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return img
