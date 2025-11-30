import matplotlib.pyplot as plt
import skimage as skim
import numpy as np

# I = plt.imread('./MEDIA/IMG/video_003_008.jpg')

# plt.subplot(6,2,1)
# plt.imshow(I, cmap='gray')

# I_gray = skim.color.rgb2gray(I)

# I_inv = skim.util.invert(I_gray)
# plt.subplot(6,2,2)
# plt.imshow(I_inv, cmap='gray')

# I_inv_corrigee = skim.restoration.denoise_tv_chambolle(I_inv)
# plt.subplot(6,2,3)
# plt.imshow(I_inv_corrigee, cmap='gray')

# I_corrigee = skim.util.invert(I_inv_corrigee)
# plt.subplot(6,2,4)
# plt.imshow(I_corrigee, cmap='gray')

# I_gamma = skim.exposure.adjust_gamma(I_corrigee, 0.5)
# plt.subplot(6,2,5)
# plt.imshow(I_gamma, cmap='gray')

# I_hist = skim.exposure.equalize_hist(I_gamma)
# plt.subplot(6,2,6)
# plt.imshow(I_hist, cmap='gray')


# skim.io.imsave('test.jpg', skim.util.img_as_ubyte(I_gamma))

# plt.show()


def ameliorer_image(img_path: str, afficher_img: bool = False) -> np.ndarray:
    """
    Prend une image (NumPy array), applique:
      - conversion en niveaux de gris
      - inversion
      - débruitage TV
      - inversion
      - correction gamma
      - égalisation d'histogramme
    et retourne l'image corrigée (float dans [0,1]).
    """

    # Convertit en niveaux de gris si l'image est couleur
    # Pipeline simple
    I = plt.imread(img_path)

    # Vérifier si l'image est en couleur ou en niveaux de gris
    if len(I.shape) == 3:  # Image couleur (RGB)
        img = skim.color.rgb2gray(I)
    else:  # Image déjà en niveaux de gris
        img = I

    I_inv = skim.util.invert(img)
    I_inv_corrigee = skim.restoration.denoise_tv_chambolle(I_inv)
    I_corrigee = skim.util.invert(I_inv_corrigee)
    I_gamma = skim.exposure.adjust_gamma(I_corrigee, 0.5)
    I_hist = skim.exposure.equalize_hist(I_gamma)

    if afficher_img:
        plt.subplot(2, 4, 1)
        plt.imshow(I, cmap="gray")
        plt.title("Image originale")
        plt.subplot(2, 4, 2)
        plt.imshow(img, cmap="gray")
        if len(I.shape) == 3:
            plt.title("Niveaux de gris")
            plt.subplot(2, 4, 2)
        else:
            plt.imshow(img, cmap="gray")
            plt.title("Image (déjà en gris)")
        plt.subplot(2, 4, 3)
        plt.imshow(I_inv, cmap="gray")
        plt.title("Inversée")
        plt.subplot(2, 4, 4)
        plt.imshow(I_inv_corrigee, cmap="gray")
        plt.title("Débruitée (TV)")
        plt.subplot(2, 4, 5)
        plt.imshow(I_corrigee, cmap="gray")
        plt.title("Ré-inversée")
        plt.subplot(2, 4, 6)
        plt.imshow(I_gamma, cmap="gray")
        plt.title("Correction gamma")
        plt.subplot(2, 4, 7)
        plt.imshow(I_hist, cmap="gray")
        plt.title("Égalisation histogramme")
        plt.show()

    return I_hist


if __name__ == "__main__":
    for i in range(1, 7):
        I_amelioree = ameliorer_image(f"./MEDIA/IMG/video_12_00{i}.jpg", True)
