import os
from ultralytics import YOLO
from PIL import Image
from ameliration_image import ameliorer_image
import numpy as np


def affichage_boxes(
    dir_path_img, nb_image: int, dir_path_obj: str = "Media/IMG/OBJ_DETECT"
):
    # Crée le dossier "IMG" s’il n’existe pas déjà
    OUTPUT_DIR = dir_path_obj
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = [f for f in os.listdir(dir_path_img) if f.endswith((".jpg", ".png"))]
    last_images = images[-nb_image:]
    # Charger le modèle
    model = YOLO("yolov8n.pt")
    # Parcourir toutes les détections
    COUNT = 0
    for image in last_images:
        # Image d'entrée
        IMAGE_PATH = f"{dir_path_img}\\{image}"

        # Prétraitement : utiliser ameliorer_image
        proc = ameliorer_image(IMAGE_PATH)  # retourne float dans [0,1]
        # convertir en uint8 3-canaux pour PIL / ultralytics
        if isinstance(proc, np.ndarray):
            proc_u8 = (np.clip(proc, 0.0, 1.0) * 255).astype(np.uint8)
            if proc_u8.ndim == 2:  # niveau de gris -> RGB
                proc_rgb = np.stack([proc_u8] * 3, axis=-1)
            elif proc_u8.ndim == 3 and proc_u8.shape[2] == 4:  # RGBA -> RGB
                proc_rgb = proc_u8[..., :3]
            else:
                proc_rgb = proc_u8
        else:
            # fallback : ouvrir l'image d'origine
            proc_rgb = np.array(Image.open(IMAGE_PATH).convert("RGB"), dtype=np.uint8)

        # Inférence sur l'image prétraitée (ndarray)
        results = model(proc_rgb)[0]
        results.show()

        # Préparer image PIL pour les découpes (utiliser l'image traitée)
        img = Image.fromarray(proc_rgb)

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ["couch", "bed", "pizza"]:
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


if __name__ == "__main__":
    affichage_boxes("MEDIA/IMG", 12)
