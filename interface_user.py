"""Module pour gérer les prompts interactifs de configuration."""

import os


def prompt_float(message, default):
    """Demande une valeur float à l'utilisateur."""
    while True:
        response = input(f"{message} (défaut: {default}): ").strip()
        if not response:
            return default
        try:
            return float(response)
        except ValueError:
            print("❌ Valeur invalide. Veuillez entrer un nombre.")


def prompt_int(message, default):
    """Demande une valeur int à l'utilisateur."""
    while True:
        response = input(f"{message} (défaut: {default}): ").strip()
        if not response:
            return default
        try:
            return int(response)
        except ValueError:
            print("❌ Valeur invalide. Veuillez entrer un nombre entier.")


def prompt_string(message, default):
    """Demande une chaîne de caractères à l'utilisateur."""
    response = input(f"{message} (défaut: {default}): ").strip()
    return response if response else default


def get_next_video_name(video_dir="Media/VID"):
    """Génère le prochain nom de vidéo basé sur le nombre de fichiers existants."""
    os.makedirs(video_dir, exist_ok=True)

    # Compter les fichiers vidéo existants
    existing_videos = [
        f for f in os.listdir(video_dir) if f.endswith((".avi", ".mp4", ".mkv"))
    ]
    next_number = len(existing_videos) + 1

    return f"video_{next_number}.avi"


def prompt_optional_string(message, default=None):
    """Demande une chaîne optionnelle à l'utilisateur."""
    prompt_text = f"{message}"
    if default:
        prompt_text += f" (défaut: {default}, Entrée pour accepter)"
    else:
        prompt_text += " (Entrée pour auto)"

    response = input(f"{prompt_text}: ").strip()
    return response if response else default


def get_capture_config():
    """Récupère la configuration via des prompts interactifs."""
    os.system("cls")
    print("\n" + "=" * 60)
    print("🎥 CONFIGURATION DE L'ENREGISTREMENT VIDÉO")
    print("=" * 60 + "\n")

    config = {}

    print("📹 Paramètres de capture:")
    config["duration"] = prompt_float("  Durée de l'enregistrement (secondes)", 30.0)
    config["photo_interval"] = prompt_float(
        "  Intervalle entre les photos (secondes)", 5.0
    )

    print("\n📁 Fichiers et dossiers:")
    default_video_name = get_next_video_name()
    config["video_out"] = prompt_optional_string(
        "  Nom du fichier vidéo (sans chemin)", default_video_name
    )
    config["photos_dir"] = prompt_string("  Dossier pour les photos", "MEDIA/IMG")

    config["camera"] = 0
    config["fps"] = 30.0
    config["codec"] = "XVID"
    config["backend"] = "auto"

    print("\n" + "=" * 60)
    print("✅ Configuration terminée!")
    print("=" * 60 + "\n")

    return config


def confirm_config(config):
    """Affiche la configuration et demande confirmation."""
    print("📋 Récapitulatif de la configuration:")
    print(f"  • Durée: {config['duration']}s")
    print(f"  • Intervalle photos: {config['photo_interval']}s")
    print(f"  • Fichier vidéo: {config['video_out']}")
    print(f"  • Dossier photos: {config['photos_dir']}")
    print()

    while True:
        response = (
            input("Confirmer et démarrer l'enregistrement? (o/n): ").strip().lower()
        )
        if response in ["o", "oui", "y", "yes"]:
            return True
        elif response in ["n", "non", "no"]:
            return False
        else:
            print("❌ Réponse invalide. Entrez 'o' ou 'n'.")
