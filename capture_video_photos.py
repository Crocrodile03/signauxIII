import cv2
import os
import time
import argparse
from datetime import datetime
from interface_user import get_next_video_name
from recup_canap_V3 import analyse_image


def parse_args():
    p = argparse.ArgumentParser(
        description="Enregistrer une vidéo et prendre des photos périodiques avec la webcam"
    )
    p.add_argument(
        "--duration",
        "-d",
        type=float,
        default=30.0,
        help="Durée de la vidéo en secondes (défaut: 30)",
    )
    p.add_argument(
        "--photo-interval",
        "-i",
        type=float,
        default=5.0,
        help="Intervalle en secondes entre chaque photo (défaut: 5)",
    )
    p.add_argument(
        "--video-out",
        type=str,
        default=None,
        help="Nom du fichier vidéo de sortie (dans dossier video/). Par défaut horodaté.",
    )
    p.add_argument(
        "--photos-dir",
        type=str,
        default="MEDIA/IMG",
        help="Dossier pour sauvegarder les photos (défaut: MEDIA/IMG)",
    )
    p.add_argument(
        "--camera", type=int, default=0, help="Index de la caméra (défaut: 0)"
    )
    p.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS pour l'enregistrement vidéo (essayer 20-30)",
    )
    p.add_argument(
        "--codec",
        type=str,
        default="XVID",
        help="Codec fourcc pour la vidéo (ex: XVID, MJPG)",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        help="Backend OpenCV à utiliser (auto, dshow, msmf, vfw, ffmpeg)",
    )
    return p.parse_args()


def open_camera(index, backend):
    backend_map = {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "vfw": cv2.CAP_VFW,
        "ffmpeg": cv2.CAP_FFMPEG,
    }
    if backend != "auto":
        flag = backend_map.get(backend.lower())
        if flag is not None:
            return cv2.VideoCapture(index, flag)
        else:
            print(f"Backend inconnu '{backend}', utilisation automatique.")
    try:
        import platform

        if platform.system().lower() == "windows":
            return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    except Exception:
        pass
    return cv2.VideoCapture(index)


def main():
    args = parse_args()

    video_dir = os.path.join("Media", "VID")
    photos_dir = args.photos_dir
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(photos_dir, exist_ok=True)

    cap = open_camera(args.camera, args.backend)
    if not cap.isOpened():
        print(
            f"Erreur: impossible d'ouvrir la caméra {args.camera} (backend={args.backend})."
        )
        return

    # Dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # FPS: préférer la valeur fournie, mais tenter de lire la valeur de la caméra
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = args.fps if args.fps and args.fps > 0 else (cam_fps if cam_fps > 0 else 20.0)

    if args.video_out:
        video_path = os.path.join(video_dir, args.video_out)
    else:
        video_name = get_next_video_name()
        video_path = os.path.join(video_dir, f"{video_name}")

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(
            "Avertissement: VideoWriter n'a pas pu s'ouvrir. Vérifiez le codec ou les permissions."
        )

    print(
        f"Enregistrement vidéo: {video_path} durée={args.duration}s fps={fps} taille=({width}x{height})"
    )
    print(f"Sauvegarde photos toutes les {args.photo_interval}s dans {photos_dir}")
    print("Appuyez sur 'q' pour interrompre prématurément.")

    start = time.time()
    last_photo = start - args.photo_interval
    saved_photos = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur: frame introuvable depuis la caméra.")
                break

            now = time.time()

            # Écrire la frame dans la vidéo
            if writer.isOpened():
                writer.write(frame)

            # Sauvegarder une photo si l'intervalle est atteint
            if now - last_photo >= args.photo_interval:
                saved_photos += 1
                # tsf = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                video_name_sans_ext = video_name.split(".")[0]
                test = video_name_sans_ext.split("_")
                test[1] = test[1].zfill(3)
                video_name_sans_ext.join("_")
                photo_name = f"{video_name_sans_ext}_{str(saved_photos).zfill(3)}.jpg"
                photo_path = os.path.join(photos_dir, photo_name)
                cv2.imwrite(photo_path, frame)
                last_photo = now
                print(f"✅ Photo sauvegardée: {photo_path} (#{saved_photos})")

            # Affichage live
            cv2.imshow("Enregistrement", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Interrompu par l'utilisateur.")
                break

            # Fin si durée atteinte
            if now - start >= args.duration:
                print(f"Durée atteinte ({args.duration}s). Fin de l'enregistrement.")
                break

    finally:
        cap.release()
        if writer.isOpened():
            writer.release()
        cv2.destroyAllWindows()
        print(f"Terminé. Vidéo: {video_path} — Photos sauvegardées: {saved_photos}")
        analyse_image("MEDIA/IMG", saved_photos)


if __name__ == "__main__":
    main()
