import cv2
import os
import time
from datetime import datetime
from interface_user import get_capture_config, confirm_config


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
    # Récupération de la configuration via prompts
    config = get_capture_config()

    # Demande de confirmation
    if not confirm_config(config):
        print("❌ Enregistrement annulé.")
        return

    video_dir = os.path.join("Media", "VID")
    photos_dir = config["photos_dir"]
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(photos_dir, exist_ok=True)

    cap = open_camera(config["camera"], config["backend"])
    if not cap.isOpened():
        print(
            f"Erreur: impossible d'ouvrir la caméra {config['camera']} (backend={config['backend']})."
        )
        return

    # Dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # FPS
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = (
        config["fps"]
        if config["fps"] and config["fps"] > 0
        else (cam_fps if cam_fps > 0 else 20.0)
    )

    if config["video_out"]:
        video_path = os.path.join(video_dir, config["video_out"])
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(video_dir, f"capture_{ts}.avi")

    fourcc = cv2.VideoWriter_fourcc(*config["codec"])
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(
            "Avertissement: VideoWriter n'a pas pu s'ouvrir. Vérifiez le codec ou les permissions."
        )

    print(f"\n🎬 Démarrage de l'enregistrement...")
    print(f"📹 Vidéo: {video_path}")
    print(f"📸 Photos: toutes les {config['photo_interval']}s dans {photos_dir}")
    print(f"⏱️  Durée: {config['duration']}s | FPS: {fps} | Taille: {width}x{height}")
    print("❗ Appuyez sur 'q' pour interrompre\n")

    start = time.time()
    last_photo = start - config["photo_interval"]
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
            if now - last_photo >= config["photo_interval"]:
                tsf = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                photo_name = f"photo_{tsf}.jpg"
                photo_path = os.path.join(photos_dir, photo_name)
                cv2.imwrite(photo_path, frame)
                saved_photos += 1
                last_photo = now
                print(f"✅ Photo sauvegardée: {photo_path} (#{saved_photos})")

            # Affichage live
            cv2.imshow("Enregistrement", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Interrompu par l'utilisateur.")
                break

            # Fin si durée atteinte
            if now - start >= config["duration"]:
                print(
                    f"\n⏱️  Durée atteinte ({config['duration']}s). Fin de l'enregistrement."
                )
                break

    finally:
        cap.release()
        if writer.isOpened():
            writer.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Terminé!")
        print(f"📹 Vidéo: {video_path}")
        print(f"📸 Photos sauvegardées: {saved_photos}")


if __name__ == "__main__":
    main()
