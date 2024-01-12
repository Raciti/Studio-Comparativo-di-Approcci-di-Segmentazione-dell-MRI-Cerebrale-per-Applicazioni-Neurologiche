
from torchvision.io import read_image
import matplotlib.pyplot as plt
from pathlib import Path

#path_img = Path("/home/rraciti/Tesi/Results/SAM_result/Masks")

img = read_image("/home/rraciti/Tesi/Results/SAM_result/Masks/002_S_1155/I995496/I995496_slice128/I995496_slice128_0.png")
img = img.permute(1,2,0)


plt.figure(figsize=(6,6))
plt.imshow(img)
plt.axis('off')
plt.show()
plt.savefig('/home/rraciti/Tesi/Codice/Code_SAM/Other/show.png')
