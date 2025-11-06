import matplotlib.pyplot as plt


img = plt.imread("IMG/lit_vide.jpg")[750:3000]
imgplot = plt.imshow(img, cmap="gray")
plt.show()