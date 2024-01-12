import math
import itertools
import torch
import torch.nn.functional as F #for resize
from pathlib import Path
import pprint
import nibabel as nib
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

import einops as E

from universeg import universeg

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)


# Fuctions
def visualize_tensors(tensors,label, col_wrap=8, col_names=None, title=None):
    M = len(tensors)
    N = len(next(iter(tensors.values())))

    cols = col_wrap
    rows = math.ceil(N/cols) * M

    d = 2.5
    fig, axes = plt.subplots(rows, cols, figsize=(d*cols, d*rows))
    if rows == 1:
      axes = axes.reshape(1, cols)

    for g, (grp, tensors) in enumerate(tensors.items()):
        for k, tensor in enumerate(tensors):
            col = k % cols
            row = g + M*(k//cols)
            x = tensor.detach().cpu().numpy().squeeze()
            ax = axes[row,col]
            if len(x.shape) == 2:
                ax.imshow(x,vmin=0, vmax=1, cmap='gray')
            else:
                ax.imshow(E.rearrange(x,'C H W -> H W C'))
            if col == 0:
                ax.set_ylabel(grp, fontsize=16)
            if col_names is not None and row == 0:
                ax.set_title(col_names[col])

    for i in range(rows):
        for j in range(cols):
            ax = axes[i,j]
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

    if title:
        plt.suptitle(title, fontsize=20)

    plt.tight_layout()
    plt.savefig(f"/home/rraciti/Tesi/Results/UniSeg/result_{label}.png")


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

def samplerShow(images, str):
    num_images = len(images)
    num_rows = 5
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
    plt.savefig(f'/home/rraciti/Tesi/Results/UniSeg/sampler_{str}.png')  # Ottimizzare lo spaziamento tra sottoplot
    plt.show()


import skimage

def brain_cropping(scan, margin, threshold_function=skimage.filters.threshold_otsu):
    '''
    Crop the brain inside the scan, minimizing the amount of black background. 
    The cropping can be amortized using a margin. The standard thresholding function
    is Otsu, but it can be changed with any other skimage.filters.thresholds
    '''
    t = threshold_function(scan)
    binary_image = scan > t
    h, w = binary_image.shape

    for right_to_left in range(0, w):
        if any(binary_image[:, right_to_left]): break

    for left_to_right in range(w-1, -1, -1):
        if any(binary_image[:, left_to_right]): break

    for top_to_bottom in range(0, h):
        if any(binary_image[top_to_bottom, :]): break

    for bottom_to_top in range(h-1, -1, -1):
        if any(binary_image[bottom_to_top, :]): break
            
    r_anchor = right_to_left - margin if right_to_left - margin > 0 else right_to_left # type: ignore
    l_anchor = left_to_right + margin if left_to_right + margin < w else left_to_right # type: ignore
    t_anchor = top_to_bottom - margin if top_to_bottom - margin > 0 else top_to_bottom # type: ignore
    b_anchor = bottom_to_top + margin if bottom_to_top + margin < h else bottom_to_top # type: ignore 

    return scan[t_anchor:(b_anchor+1), r_anchor:(l_anchor+1)]


def square_padding(scan, resize_to=None):
    '''
    Pads the lowest dimension of the image to make it shaped
    like a square. Eventually, the output image can be reshaped
    using the `resize_to` argument.  
    '''
    h, w = scan.shape
    max_dim = max(scan.shape)
    output_shape_placeholder = np.zeros((max_dim, max_dim))
    top_left_x = int(np.round((max_dim - h) / 2))
    top_left_y = int(np.round((max_dim - w) / 2))
    output_shape_placeholder[top_left_x:(top_left_x + h), top_left_y:(top_left_y + w)] = scan.copy()
    if resize_to is not None:
        return skimage.transform.resize(output_shape_placeholder, resize_to)
    return output_shape_placeholder

import os

def main(_label):
    dataset_dir = r'/home/rraciti/Tesi/Dataset/Ravì'

    support_images, support_labels = [], []
    label = _label

    for patient in os.listdir(dataset_dir):
        if patient[0] == '.': continue
        
        # load the mri and the parcellation
        patient_path = os.path.join(dataset_dir, patient, '3DT1', 'baseline_to_last_visit')
        mri = nib.load( os.path.join(patient_path, 'A.nii.gz') )
        #seg = nib.load( os.path.join(patient_path, 'A_to_B_edgepoints.nii.gz') )
        seg = nib.load( os.path.join(patient_path, 'A_parc.nii.gz') )
        
        # get the central slice 
        mri_arr = mri.get_fdata()
        seg_arr = seg.get_fdata()
        seg_arr[ seg_arr != label ] = 0
        seg_arr[ seg_arr == label ] = 1
        
        assert mri_arr.shape == seg_arr.shape, 'shape does not match'

        for i in range(-2, 3):
            mri_slice = mri_arr[ :, :, mri_arr.shape[2] // 2 + i ]
            seg_slice = seg_arr[ :, :, seg_arr.shape[2] // 2 + i ]
            # adapting the size
            mri_slice = square_padding(mri_slice, resize_to=(128, 128))
            seg_slice = square_padding(seg_slice, resize_to=(128, 128))

            mri_slice = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min())
            seg_slice = (seg_slice - seg_slice.min()) / (seg_slice.max() - seg_slice.min())

            support_images.append(torch.from_numpy(mri_slice).unsqueeze(0))
            support_labels.append(torch.from_numpy(seg_slice).unsqueeze(0))

            # mri_slice = (255 * (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min())).astype(np.uint8)
            # seg_slice = (255 * (seg_slice - seg_slice.min()) / (seg_slice.max() - seg_slice.min())).astype(np.uint8)
            # skimage.io.imsave('mri.png', mri_slice)
            # skimage.io.imsave('seg.png', seg_slice)

    samplerShow(support_labels, f"label_{label}")
    #samplerShow(support_images, "image")

    image_batch = support_images[19].unsqueeze(0).to(device).to(torch.float32)
    image_label = support_labels[19].unsqueeze(0).to(device)

    s_image_batch = torch.stack(support_images[:18]).unsqueeze(0).to(device).to(torch.float32)
    s_label_batch = (torch.stack(support_labels[:18]).unsqueeze(0) == True).to(device)

    model = universeg(pretrained=True)
    model.to(device = device)

    # run inference
    logits = model(image_batch, s_image_batch, s_label_batch)[0].to('cpu')
    pred = torch.sigmoid(logits)

    res = {'data': [image_batch, image_label, pred, pred > 0.5]}
    titles = col_names=['image', 'label', 'pred (soft)', 'pred (hard)']
    visualize_tensors(res, label=label, col_wrap=4, col_names=titles)
    torch.cuda.empty_cache()

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Programma di esempio con argparse')
    
    parser.add_argument('label',type= int, help= "Atrophy label. Value must be between 1 and 19")
    
    args = parser.parse_args()

    try:
        if 1 <= args.label <= 19:  
            main(args.label)  
        else:
            print("The input is not between in the allowed range (1-19)")
    except argparse.ArgumentTypeError as e:
        print(e)
