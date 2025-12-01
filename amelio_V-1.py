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

def ameliorer_image(img: np.ndarray) -> np.ndarray:
    """
    Prend une image (NumPy array), applique:
      - conversion en niveaux de gris
      - inversion
      - débruitage TV
      - inversion
      - correction gamma (auto)
      - égalisation d'histogramme
    et retourne l'image corrigée (float dans [0,1]).
    """
    # Convertit en niveaux de gris si l'image est couleur
    img = skim.color.rgb2gray(img)

    # Pipeline simple
    I_inv = skim.util.invert(img)
    I_inv_corrigee = skim.restoration.denoise_tv_chambolle(I_inv)
    I_corrigee = skim.util.invert(I_inv_corrigee)

    # Gamma automatique en fonction de la luminance moyenne (img en [0,1])
    mean_luma = float(np.mean(I_corrigee))
    # seuils ~110/255 et ~150/255 convertis en [0,1]
    low_thr, high_thr = 110 / 255.0, 150 / 255.0
    if mean_luma < low_thr:
        gamma = float(np.interp(mean_luma, [0.16, low_thr], [0.6, 0.95]))   # éclaircir
    elif mean_luma > high_thr:
        gamma = float(np.interp(mean_luma, [high_thr, 0.90], [1.05, 1.6]))  # assombrir
    else:
        gamma = 1.0

    I_gamma = skim.exposure.adjust_gamma(I_corrigee, gamma)
    I_hist = skim.exposure.equalize_hist(I_gamma)

    return I_hist

if __name__ == "__main__":
    I = plt.imread('./MEDIA/IMG/video_003_008.jpg')
    I_amelioree = ameliorer_image(I)
    plt.imshow(I_amelioree, cmap='gray')
    plt.show()