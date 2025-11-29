import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage import img_as_ubyte


def eclaircissement_image(
    img_path: str, method: str = "clahe", clip_limit: float = 0.03, gamma: float = 1.0
):
    """
    Éclaircit une image sans la surexposer :
    - method="clahe" : contraste local (recommended)
    - method="gamma" : correction gamma simple
    """
    I = plt.imread(img_path)

    # convertir en float [0,1] si nécessaire
    if I.dtype != np.float32 and I.dtype != np.float64:
        I = I.astype("float32") / 255.0

    # séparer alpha si présent
    alpha = None
    if I.ndim == 3 and I.shape[2] == 4:
        alpha = I[..., 3]
        I = I[..., :3]

    # choisir la méthode
    if method == "clahe":
        if I.ndim == 2:  # niveaux de gris
            I_out = exposure.equalize_adapthist(I, clip_limit=clip_limit)
        else:  # couleur : appliquer par canal (évite artefacts)
            I_out = np.stack(
                [
                    exposure.equalize_adapthist(I[..., c], clip_limit=clip_limit)
                    for c in range(I.shape[2])
                ],
                axis=-1,
            )
    elif method == "gamma":
        I_out = exposure.adjust_gamma(I, gamma=gamma)
    else:
        # fallback : rescale simple + optional gamma
        I_out = exposure.rescale_intensity(I, in_range="image", out_range=(0.0, 1.0))
        if gamma != 1.0:
            I_out = exposure.adjust_gamma(I_out, gamma=gamma)

    # optionnel : légère correction gamma pour éviter effet trop clair
    if method == "clahe" and gamma != 1.0:
        I_out = exposure.adjust_gamma(I_out, gamma=gamma)

    # remettre canal alpha si nécessaire
    if alpha is not None:
        I_out = np.dstack([I_out, alpha])

    return img_as_ubyte(I_out)
