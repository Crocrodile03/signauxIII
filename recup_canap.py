import os
from ultralytics import YOLO
from PIL import Image
import cv2


def affichage_boxes(
    dir_path_img,
    nb_image: int = 10,
    dir_path_obj: str = "Media/IMG/OBJ_DETECT",
    use_preprocessing: bool = False,  # Désactivé par défaut
):
    """
    Détecte et extrait les objets (lit, canapé) des images.

    Args:
        dir_path_img: Dossier contenant les images
        nb_image: Nombre d'images à traiter
        dir_path_obj: Dossier de sortie pour les objets détectés
        use_preprocessing: Si False, utilise l'image originale (recommandé)
    """
    OUTPUT_DIR = dir_path_obj
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = [f for f in os.listdir(dir_path_img) if f.endswith((".jpg", ".png"))]
    last_images = images[-nb_image:]

    model = YOLO("yolov8n.pt")
    COUNT = 0

    for image in last_images:
        IMAGE_PATH = os.path.join(dir_path_img, image)

        # Lire l'image directement (sans traitement supplémentaire)
        img_bgr = cv2.imread(IMAGE_PATH)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Inférence YOLO
        results = model(img_rgb, verbose=False)[0]
        results.show()
        # Convertir en PIL pour les découpes
        img_pil = Image.fromarray(img_rgb)

        # Extraire les objets détectés
        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ["couch", "bed"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img_pil.crop((x1, y1, x2, y2))

                # Créer le sous-dossier pour l'objet
                obj_dir = os.path.join(OUTPUT_DIR, label)
                os.makedirs(obj_dir, exist_ok=True)

                # Sauvegarder l'objet extrait
                COUNT += 1
                filename = os.path.join(obj_dir, f"{label}_{COUNT}.jpg")
                crop.save(filename)
                print(f"✅ {label.capitalize()} détecté et sauvegardé : {filename}")

    if COUNT == 0:
        print("❌ Aucun canapé ni lit détecté.")
    else:
        print(f"🎉 {COUNT} objet(s) détecté(s) et sauvegardé(s) dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    affichage_boxes("MEDIA/IMG", nb_image=12, use_preprocessing=False)
