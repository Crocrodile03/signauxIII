import numpy as np
import cv2

def ameliorer_image(
    img: np.ndarray,
    strength: float = 1.0,
    return_uint8: bool = True,
    assume_bgr: bool = False,
    good_luma_range: tuple[int, int] = (110, 165),  # plage considérée “correcte”
) -> np.ndarray:
    """
    Si la luminosité moyenne est dans good_luma_range -> pas de traitement.
    Sinon, applique la chaîne d'amélioration actuelle.
    """
    if img is None:
        raise ValueError("Image vide")

    # Assure RGB uint8
    if assume_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype != np.uint8:
        img_max = float(np.max(img)) if img.size else 1.0
        img = np.clip(img * (255.0 if img_max <= 1.0 else 1.0), 0, 255).astype(np.uint8)

    # Test de luminosité (moyenne en niveaux de gris)
    gray_mean = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean()
    if good_luma_range[0] <= gray_mean <= good_luma_range[1]:
        # Retourne l'image telle quelle
        return img if return_uint8 else (img.astype(np.float32) / 255.0)

    # ----- Pipeline existant (inchangé) -----

    # 1) White balance (gray-world)
    mean = img.reshape(-1, 3).mean(axis=0, keepdims=True)  # R,G,B moyennes
    gray = float(mean.mean())
    gain = gray / (mean + 1e-6)
    wb = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    # 2) CLAHE sur L dans LAB
    lab = cv2.cvtColor(wb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clip = 2.0 * float(np.clip(strength, 0.3, 3.0))  # plus fort => plus de contraste local
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    img_clahe = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB)

    # 3) Dénmage léger (conserve les détails)
    h = 3 + int(3 * strength)
    den = cv2.fastNlMeansDenoisingColored(img_clahe, None, h, h, 7, 21)

    # 4) Netteté (unsharp mask)
    sigma = 1.0 + 0.7 * strength
    blur = cv2.GaussianBlur(den, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(den, 1.0 + 0.6 * strength, blur, -0.6 * strength, 0)

    # 5) Gamma auto selon la luminance moyenne
    gray_mean = cv2.cvtColor(sharp, cv2.COLOR_RGB2GRAY).mean()
    if gray_mean < 110:
        gamma = float(np.interp(gray_mean, [40, 110], [0.6, 0.95]))  # éclaircir
    elif gray_mean > 150:
        gamma = float(np.interp(gray_mean, [150, 230], [1.05, 1.6]))  # assombrir
    else:
        gamma = 1.0
    norm = np.clip(sharp.astype(np.float32) / 255.0, 0, 1)
    gamma_img = np.clip((norm ** gamma) * 255.0, 0, 255).astype(np.uint8)

    # 6) Étirement de contraste doux (1–99 percentiles, par canal)
    def stretch(x: np.ndarray) -> np.ndarray:
        lo = np.percentile(x, 1)
        hi = np.percentile(x, 99)
        if hi <= lo + 1e-6:
            return x
        y = (x.astype(np.float32) - lo) * (255.0 / (hi - lo))
        return np.clip(y, 0, 255).astype(np.uint8)

    final = np.dstack([stretch(gamma_img[..., c]) for c in range(3)])

    return final if return_uint8 else (final.astype(np.float32) / 255.0)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage import data

    # Image de test
    img = plt.imread("./MEDIA/IMG/video_003_008.jpg")  # RGB uint8

    improved = ameliorer_image(img, strength=1.5, return_uint8=True)

    # Affichage
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Originale")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Améliorée")
    plt.imshow(improved)
    plt.axis("off")

    plt.tight_layout()
    plt.show()