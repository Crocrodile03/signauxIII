import matplotlib.pyplot as plt
import skimage as skim
import numpy as np
import cv2
from skimage.filters import median
from skimage.morphology import square

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

def _show_steps(steps: dict) -> None:
    names = list(steps.keys())
    imgs = list(steps.values())
    cols = 4
    rows = int(np.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, name, im in zip(axes, names, imgs):
        ax.imshow(im, cmap='gray')
        ax.set_title(name)
        ax.axis('off')
    for ax in axes[len(imgs):]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def ameliorer_image(img: np.ndarray) -> dict[str, np.ndarray]:
    """
    Pipeline: gray -> hist_eq -> denoise -> median1 -> median2.
    Retourne toutes les étapes (float [0,1]) et les affiche.
    """
    # 1) Gris
    I_gray = skim.color.rgb2gray(img)

    # 2) Equalize histogram
    I_hist = skim.exposure.equalize_hist(I_gray)

    # 3) Denoise (TV)
    I_dn1 = skim.restoration.denoise_tv_chambolle(I_hist)

    # 4) Deux filtrages médians
    I_med1 = median(I_dn1, square(3))
    I_med2 = median(I_med1, square(3))

    steps = {
        "original_gray": I_gray,
        "hist_eq": I_hist,
        "denoise1": I_dn1,
        "median1": I_med1,
        "median2": I_med2,
    }

    _show_steps(steps)
    return steps




if __name__ == "__main__":
    I = plt.imread("./MEDIA/IMG/video_029_004.jpg")
    steps = ameliorer_image(I)
    final = steps["median4"]