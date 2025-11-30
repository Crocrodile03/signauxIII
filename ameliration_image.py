import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Tuple, Optional


def ameliorer_image(
    img_path: str, afficher_img: bool = False, retourner_uint8: bool = False
) -> np.ndarray:
    """
    Pré-traite une image pour la détection YOLO (MINIMAL - préserve les features).

    Args:
        img_path: Chemin vers l'image
        afficher_img: Afficher les étapes de traitement
        retourner_uint8: Si True, retourne uint8 [0-255], sinon float32 [0-1]

    Returns:
        Image améliorée (RGB)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_original = img_rgb.copy()

    # Étape 1: Correction légère de la balance des blancs
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)

    # Réduire modérément la dominante bleue
    a_mean = np.mean(a)
    b_mean = np.mean(b)

    a_corrected = cv2.add(a, int(128 - a_mean) // 4)  # Correction modérée
    b_corrected = cv2.subtract(b, int(b_mean - 128) // 4)  # Correction modérée

    img_wb = cv2.merge([l, a_corrected, b_corrected])
    img_wb_rgb = cv2.cvtColor(img_wb, cv2.COLOR_LAB2RGB)

    # Étape 2: Correction gamma très légère (seulement si trop clair)
    gray = cv2.cvtColor(img_wb_rgb, cv2.COLOR_RGB2GRAY)
    moyenne_luminosite = np.mean(gray)

    if moyenne_luminosite > 160:  # Image très surexposée
        gamma = 1.3
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
            "uint8"
        )
        img_gamma = cv2.LUT(img_wb_rgb, table)
    else:
        img_gamma = img_wb_rgb

    # Étape 3: CLAHE très léger (juste pour les zones sombres)
    img_lab2 = cv2.cvtColor(img_gamma, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab2)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)

    img_clahe = cv2.merge([l_clahe, a_channel, b_channel])
    img_final = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)

    # Clip des valeurs
    img_final = np.clip(img_final, 0, 255).astype(np.uint8)

    if afficher_img:
        _afficher_pipeline(
            img_original, img_wb_rgb, img_gamma, img_final, moyenne_luminosite
        )

    if retourner_uint8:
        return img_final
    return img_final.astype(np.float32) / 255.0


def _afficher_pipeline(
    img_original, img_wb_rgb, img_gamma, img_final, moyenne_luminosite
):
    """Affiche le pipeline de traitement"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].imshow(img_original)
    axes[0, 0].set_title("1. Image originale")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_wb_rgb)
    axes[0, 1].set_title("2. Balance des blancs")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img_gamma)
    axes[0, 2].set_title(f"3. Correction gamma\n(Lum: {moyenne_luminosite:.1f})")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(img_final)
    axes[1, 0].set_title("4. CLAHE léger + Final")
    axes[1, 0].axis("off")

    # Histogramme
    axes[1, 1].hist(
        cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY).ravel(),
        bins=256,
        color="blue",
        alpha=0.5,
        label="Original",
    )
    axes[1, 1].hist(
        cv2.cvtColor(img_final, cv2.COLOR_RGB2GRAY).ravel(),
        bins=256,
        color="red",
        alpha=0.5,
        label="Traité",
    )
    axes[1, 1].set_title("Histogramme")
    axes[1, 1].legend()

    # Différence
    diff = cv2.absdiff(img_original, img_final)
    axes[1, 2].imshow(diff)
    axes[1, 2].set_title("Différence")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


def detecter_flou(img_path: str) -> Tuple[float, str]:
    """Détecte le niveau de flou d'une image"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de lire l'image: {img_path}")

    variance = cv2.Laplacian(img, cv2.CV_64F).var()

    if variance > 500:
        statut = "nette"
    elif variance > 200:
        statut = "légèrement floue"
    elif variance > 100:
        statut = "floue"
    else:
        statut = "très floue"

    return variance, statut


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        img_path = sys.argv[1]

        variance, statut = detecter_flou(img_path)
        print(f"Analyse de flou: {statut} (variance: {variance:.2f})")

        img_amelioree = ameliorer_image(
            img_path, afficher_img=True, retourner_uint8=True
        )

        output_path = img_path.replace(".", "_amelioree.")
        cv2.imwrite(output_path, cv2.cvtColor(img_amelioree, cv2.COLOR_RGB2BGR))
        print(f"Image sauvegardée: {output_path}")
    else:
        print("Usage: python ameliration_image.py <chemin_image>")
