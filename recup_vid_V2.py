import cv2
from ultralytics import YOLO
from datetime import datetime
import csv
import os

# --- Fonction utilitaire ---
def boxes_intersect(boxA, boxB):
    """Renvoie True si deux boxes se chevauchent"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    return inter_width * inter_height > 0

# --- Charger le modèle YOLO ---
model = YOLO("yolov8n.pt")

# --- Ouvrir la webcam ---
cap = cv2.VideoCapture(0)

# --- Paramètres du filtre temporel ---
frames_needed = 5
presence_counter = 0
absence_counter = 0
person_on_couch_state = False  # état stable (True/False)

# --- Préparer le fichier CSV ---
log_file = "historique_presence.csv"
write_header = not os.path.exists(log_file)

with open(log_file, mode="a", newline="") as file:
    writer = csv.writer(file)
    if write_header:
        writer.writerow(["Heure", "Événement"])  # En-tête

# --- Boucle principale ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    persons, couches = [], []

    # --- Extraire les boxes ---
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "person":
            persons.append((x1, y1, x2, y2))
        elif label == "couch":
            couches.append((x1, y1, x2, y2))

    # --- Vérifier intersection ---
    detected_now = any(
        boxes_intersect(couch, person) for couch in couches for person in persons
    )

    # --- Appliquer le filtrage temporel ---
    if detected_now:
        presence_counter += 1
        absence_counter = 0
    else:
        absence_counter += 1
        presence_counter = 0

    event = None  # Pour savoir si un changement d'état a eu lieu

    if presence_counter >= frames_needed and not person_on_couch_state:
        person_on_couch_state = True
        event = "🟢 Personne assise"
    elif absence_counter >= frames_needed and person_on_couch_state:
        person_on_couch_state = False
        event = "🔴 Personne partie"

    # --- Enregistrer les événements ---
    if event:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - {event}")

        # Sauvegarde dans le CSV
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, event])

    # --- Affichage ---
    color = (0, 255, 0) if person_on_couch_state else (0, 0, 255)
    text = "Personne sur le canapé" if person_on_couch_state else "Canapé vide"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    cv2.imshow("Détection de présence", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Échap pour quitter
        break

cap.release()
cv2.destroyAllWindows()
