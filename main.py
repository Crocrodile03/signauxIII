import cv2
import os
import time
import argparse
from pathlib import Path
from ultralytics import YOLO
from interface_user import get_next_video_name
from detection_person import analyse_image, dessiner_detections, DetectionResult
from recup_canap import affichage_boxes
from typing import List


def parse_args():
    p = argparse.ArgumentParser(description="Système de surveillance de chute")
    p.add_argument(
        "--duration",
        "-d",
        type=float,
        default=30.0,
        help="Durée en secondes (défaut: 30)",
    )
    p.add_argument(
        "--photo-interval",
        "-i",
        type=float,
        default=5.0,
        help="Intervalle entre photos en secondes (défaut: 5)",
    )
    p.add_argument(
        "--photos-dir", type=str, default="MEDIA/IMG", help="Dossier des photos"
    )
    p.add_argument(
        "--detections-dir",
        type=str,
        default="MEDIA/DETECTIONS",
        help="Dossier pour les images avec détections",
    )
    p.add_argument("--camera", type=int, default=0, help="Index caméra")
    p.add_argument("--fps", type=float, default=30.0, help="FPS vidéo")
    p.add_argument(
        "--no-display", action="store_true", help="Ne pas afficher la fenêtre live"
    )
    p.add_argument(
        "--resolution",
        type=str,
        default="auto",
        help="Résolution (auto, 640x480, 1280x720, 1920x1080)",
    )
    return p.parse_args()


def configure_camera(cap, resolution: str = "auto"):
    """
    Configure la caméra avec la meilleure résolution

    Args:
        cap: Objet VideoCapture
        resolution: Résolution souhaitée (auto, 640x480, 1280x720, 1920x1080)

    Returns:
        tuple: (width, height, fps)
    """
    # Résolutions communes
    resolutions = {
        "1920x1080": (1920, 1080),
        "1280x720": (1280, 720),
        "640x480": (640, 480),
        "320x240": (320, 240),
    }

    if resolution != "auto" and resolution in resolutions:
        w, h = resolutions[resolution]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    # Lire les dimensions actuelles de la caméra
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Si FPS invalide, utiliser valeur par défaut
    if fps <= 0 or fps > 120:
        fps = 30.0

    print(f"📷 Résolution caméra: {width}x{height} @ {fps:.1f} FPS")

    # Tester en lisant une frame
    ret, test_frame = cap.read()
    if ret:
        actual_height, actual_width = test_frame.shape[:2]
        if actual_width != width or actual_height != height:
            print(f"⚠️  Résolution réelle détectée: {actual_width}x{actual_height}")
            width, height = actual_width, actual_height

    return width, height, fps


def main():
    args = parse_args()

    # Créer les dossiers
    video_dir = Path("Media/VID")
    photos_dir = Path(args.photos_dir)
    detections_dir = Path(args.detections_dir)
    use_preprocessing = True

    for d in [video_dir, photos_dir, detections_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Initialiser YOLO une seule fois
    print("🔄 Chargement du modèle YOLO...")
    model = YOLO("yolov8n.pt")
    print("✅ Modèle chargé")

    # Ouvrir la caméra
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"❌ Erreur: impossible d'ouvrir la caméra {args.camera}")
        return

    # Configurer la caméra et obtenir les vraies dimensions
    width, height, fps = configure_camera(cap, args.resolution)

    # Utiliser le FPS des arguments si spécifié, sinon celui de la caméra
    if args.fps != 30.0:
        fps = args.fps

    # Préparer l'enregistrement vidéo
    video_name = get_next_video_name()
    video_path = video_dir / video_name
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    print(f"\n{'='*60}")
    print(f"🎥 Enregistrement: {video_path}")
    print(f"📐 Dimensions: {width}x{height} @ {fps:.1f} FPS")
    print(f"⏱️  Durée: {args.duration}s | 📸 Intervalle photos: {args.photo_interval}s")
    print(f"{'='*60}\n")

    start = time.time()
    last_photo = start - args.photo_interval
    saved_photos = 0
    detection_results: List[DetectionResult] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Erreur lecture frame")
                saved_photos = 5
                return

            now = time.time()

            # Vérifier que la frame a les bonnes dimensions
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            writer.write(frame)

            # Capturer et analyser photo
            if now - last_photo >= args.photo_interval:
                saved_photos += 1
                video_base = video_name.split(".")[0]
                parts = video_base.split("_")
                parts[1] = parts[1].zfill(3)
                photo_name = f"{'_'.join(parts)}_{str(saved_photos).zfill(3)}.jpg"
                photo_path = photos_dir / photo_name

                cv2.imwrite(str(photo_path), frame)
                last_photo = now

                # Analyser l'image
                print(f"\n📸 Analyse de {photo_name}...")
                result = analyse_image(
                    str(photo_path),
                    model=model,
                    verbose=True,
                    use_preprocessing=use_preprocessing,
                    iou_threshold=0.2,
                    overlap_threshold=0.4,
                )
                detection_results.append(result)

                # Dessiner les détections
                detection_path = detections_dir / f"detection_{photo_name}"
                dessiner_detections(str(photo_path), result, str(detection_path))

                if result.status_message:
                    status_emoji = "⚠️"
                elif result.person_in_bed:
                    status_emoji = "✅"
                else:
                    status_emoji = "🚨"
                print(f"{status_emoji} {result}")

            # Affichage live
            if not args.no_display:
                elapsed = int(now - start)
                remaining = int(args.duration - elapsed)

                # Frame pour affichage (clone pour ne pas modifier l'original)
                display_frame = frame.copy()

                cv2.putText(
                    display_frame,
                    f"Temps restant: {remaining}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display_frame,
                    f"{width}x{height}",
                    (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow("Surveillance", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n⚠️ Interrompu par l'utilisateur")
                    break

            if now - start >= args.duration:
                print(f"\n✅ Durée atteinte ({args.duration}s)")
                break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        # Rapport final
        print(f"\n{'='*60}")
        print("📊 RÉSUMÉ DES DÉTECTIONS")
        print(f"{'='*60}")

        chutes = 0
        avertissements = 0
        ok = 0

        for result in detection_results:
            print(result)
            if result.status_message:
                avertissements += 1
            elif not result.person_in_bed:
                chutes += 1
            else:
                ok += 1

        print(f"{'='*60}")
        print(f"📹 Vidéo: {video_path}")
        print(f"📸 Photos: {saved_photos}")
        print(f"✅ OK: {ok} | 🚨 Chutes: {chutes} | ⚠️ Avertissements: {avertissements}")
        print(f"{'='*60}\n")

        # Afficher les images avec détections
        print("\n🖼️ Affichage des détections...")
        for result in detection_results:
            detection_img_name = f"detection_{Path(result.image_path).name}"
            detection_img_path = detections_dir / detection_img_name

            if detection_img_path.exists():
                img = cv2.imread(str(detection_img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                cv2.imshow("Detections (touche pour suivant, 'q' pour quitter)", img)
                key = cv2.waitKey(0)  # Attendre indéfiniment

                if key == ord("q"):  # Quitter si 'q'
                    break
        cv2.destroyAllWindows()
        print("\n✅ Pipeline complet terminé!")


if __name__ == "__main__":
    main()
