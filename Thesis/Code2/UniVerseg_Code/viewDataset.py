from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F #for resize


def resize(tensor):
    # Resize the tensor using bilinear interpolation
    resized_tensor = F.interpolate(tensor.unsqueeze(0), size= (128,128), mode="bilinear", align_corners=False)

    return resized_tensor.squeeze(0)

def show(tensor, i=''):
    tensor = tensor.permute(1,2,0)
    plt.figure(figsize=(10,10))
    plt.imshow(tensor, cmap='gray')
    plt.axis('off')
    plt.savefig(f'/home/rraciti/Tesi/Codice/Code_UniverSeg/Example/show_image{i}.png')
    print(f"The plot is saved in Other")
    # Show the plot (optional)
    plt.show()



def multiShow(images, str):
    num_images = len(images)
    num_rows = 4
    num_cols = 5

    # Creazione della figura e sottoplot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i].permute(1,2,0), cmap='gray')
            ax.axis('off')  # Nascondi i numeri sugli assi

    # Nascondere gli assi dei sottoplot vuoti
    for j in range(i + 1, num_rows * num_cols):
        axes.flat[j].axis('off')

    plt.tight_layout()
    plt.savefig(f'/home/rraciti/Tesi/Codice/Code_UniverSeg/Example/test{str}.png')  # Ottimizzare lo spaziamento tra sottoplot
    plt.show()


# images = []
# dataset = Path("/home/rraciti/Tesi/Dataset/Ravì")
# for patient in dataset.iterdir():
#     path = patient / "3DT1/baseline_to_last_visit/"
#     for file in path.iterdir():
#         print(file)
#         """if "edgepoints" in str(file):
#             nifti_image = nib.load(str(file))
#             data = nifti_image.get_fdata()
#             img = data[:, :, 130][:, :, np.newaxis]
#             # img = np.repeat(img, 3, axis=2)
#             # img = 255 - (255 * (img.max() - img) / (img.max() - img.min()))
#             img = torch.from_numpy(img)
#             img = img.permute(2,1,0) #for visualize image topdown
#             # print(img.shape)
#             show(img, f"_edge")
#             img = resize(img)
#             show(img, f"_edge_re")
            

#         if "A.nii.gz" in str(file):
#             nifti_image = nib.load(str(file))
#             data = nifti_image.get_fdata()
#             print(data.shape)
#             img = data[:, :, 130][:, :, np.newaxis]
#             img = 255 - (255 * (img.max() - img) / (img.max() - img.min()))
#             img = torch.from_numpy(img)
#             img = img.permute(2,1,0)
#             show(img, f"_A")
#             img = resize(img)
#             show(img, "A_re")"""

#         if "A_parc" in str(file):
#             # nifti_image = nib.load(str(file))
#             # data = nifti_image.get_fdata()
#             # print(data.shape)
#             # img = data[:, :, 130][:, :, np.newaxis]
#             # img = 255 - (255 * (img.max() - img) / (img.max() - img.min()))
#             # img = torch.from_numpy(img)
#             # img = img.permute(2,1,0)
#             # show(img, f"_A_parc")
#             # img = resize(img)
#             # show(img, "_A_parc_re")
#             # img_pr = (img > 170) & (img < 180)
#             # show(img_pr, "_A_parc_try")
#             for slice in range(20):    
#                 nifti_image = nib.load(str(file))
#                 data = nifti_image.get_fdata()
#                 img = data[:, :, 121+ + slice][:, :, np.newaxis]
#                 img = 255 - (255 * (img.max() - img) / (img.max() - img.min()))
#                 img = torch.from_numpy(img)
#                 img = img.permute(2,1,0)
#                 img = resize(img)
#                 img = (img > 170) & (img < 180) #serve per prendere solo la parte di atrofia che ci interessa
#                 images.append(img)
#             multiShow(images, "Multiplo")

#     break

support_labels = []
result = Path("/home/rraciti/Tesi/Results/SAM_result/3D_Image")
for patient in result.iterdir():
    for file in patient.iterdir():
        print(file)
        nifti_image = torch.load(str(file))
        for slice in range(20):    
            img = nifti_image[121 + int(slice),:,:,:]
            #img = 255 - (255 * (img.max() - img) / (img.max() - img.min()))
            img = img.permute(2,1,0)
            img = resize(img)
            img = (img > 2) #serve per prendere solo la parte di atrofia che ci interessa
            support_labels.append(img)
        label =  nifti_image[140,:,:,:]
        #label = 255 - (255 * (label.max() - label) / (label.max() - label.min()))
        label = resize(label.permute(2,1,0))
        label = (label >2)
        show(label, "Pr")
        break
    break
multiShow(support_labels, "2")