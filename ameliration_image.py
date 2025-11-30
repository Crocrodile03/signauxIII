import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Tuple, Optional


def ameliorer_image(
    img_path: str, afficher_img: bool = False, retourner_uint8: bool = False
) -> np.ndarray:
    """
    Pré-traite une image pour la détection de personnes dans un lit.

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

    # Pipeline de traitement
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)

    img_clahe = cv2.merge([l_clahe, a_channel, b_channel])
    img_clahe_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)

    img_denoised = cv2.bilateralFilter(img_clahe_rgb, d=9, sigmaColor=75, sigmaSpace=75)

    gray = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2GRAY)
    moyenne_luminosite = np.mean(gray)

    if moyenne_luminosite < 100:
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
            "uint8"
        )
        img_corrected = cv2.LUT(img_denoised, table)
    else:
        img_corrected = img_denoised

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_sharpened = cv2.filter2D(img_corrected, -1, kernel)
    img_final = cv2.normalize(img_sharpened, None, 0, 255, cv2.NORM_MINMAX)

    if afficher_img:
        _afficher_pipeline(
            img_original,
            img_clahe_rgb,
            img_denoised,
            img_corrected,
            img_sharpened,
            img_final,
            moyenne_luminosite,
        )

    if retourner_uint8:
        return img_final.astype(np.uint8)
    return img_final.astype(np.float32) / 255.0


def _afficher_pipeline(
    img_original,
    img_clahe_rgb,
    img_denoised,
    img_corrected,
    img_sharpened,
    img_final,
    moyenne_luminosite,
):
    """Affiche le pipeline de traitement"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(img_original)
    axes[0, 0].set_title("Image originale")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_clahe_rgb)
    axes[0, 1].set_title("CLAHE appliqué")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img_denoised)
    axes[0, 2].set_title("Débruitage bilateral")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(img_corrected)
    axes[0, 3].set_title(f"Correction gamma\n(Lum moy: {moyenne_luminosite:.1f})")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(img_sharpened)
    axes[1, 0].set_title("Amélioration contours")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(img_final)
    axes[1, 1].set_title("Image finale")
    axes[1, 1].axis("off")

    axes[1, 2].hist(
        cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY).ravel(),
        bins=256,
        color="blue",
        alpha=0.5,
        label="Original",
    )
    axes[1, 2].hist(
        cv2.cvtColor(img_final, cv2.COLOR_RGB2GRAY).ravel(),
        bins=256,
        color="red",
        alpha=0.5,
        label="Traité",
    )
    axes[1, 2].set_title("Histogramme")
    axes[1, 2].legend()

    diff = cv2.absdiff(img_original, img_final)
    axes[1, 3].imshow(diff)
    axes[1, 3].set_title("Différence")
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.show()
