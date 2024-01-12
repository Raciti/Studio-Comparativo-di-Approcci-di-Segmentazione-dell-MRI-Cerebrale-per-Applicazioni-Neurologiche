import os
import matplotlib.pyplot as plt
from PIL import Image

folder_path = "/home/rraciti/Tesi/Results/SAM_result/Masks/002_S_1155/I995496/I995496_slice128/"

# Lista dei file presenti nella cartella
image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

# Creazione del plot
fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(15, 12))

for i, ax in enumerate(axes.flat):
    # Caricamento dell'immagine e aggiunta al plot
    img_path = os.path.join(folder_path, image_files[i])
    img = Image.open(img_path)
    ax.imshow(img)
    ax.axis('off')

# Mostra il plot
plt.tight_layout()
plt.show()

# Salva il plot come immagine
plt.savefig('/home/rraciti/Tesi/Codice/Code_SAM/Other/show.png')
