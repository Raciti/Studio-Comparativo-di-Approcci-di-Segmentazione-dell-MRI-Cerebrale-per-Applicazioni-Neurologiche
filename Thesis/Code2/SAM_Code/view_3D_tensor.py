import torch
from pathlib import Path
import matplotlib.pyplot as plt
import sys

def main(slice, patient, img):
    img = img + ".pt"
    image = Path("/home/rraciti/Tesi/Results/SAM_result/3D_Image") / patient / img
    img_3d = torch.load(str(image))
    tensor = img_3d[int(slice),:,:,:]
    torch.save(tensor, "/home/rraciti/Tesi/Codice/Code_SAM/Other/3d_slice_tensor.pt")
    plt.figure(figsize=(10,10))
    plt.title(f"Image {str(img).split('/')[-1].split('.')[0]} \n Slice: {slice}",  fontsize= 18, fontweight="bold")
    plt.imshow(tensor)
    plt.axis('off')
    plt.savefig('/home/rraciti/Tesi/Codice/Code_SAM/Other/3d_image.png')
    print(f"The plot is saved in Other")
    # Show the plot (optional)
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) == 1:
        arg1 = "128" # slice number
        arg2 = "002_S_1155" # Id Patien
        arg3 = "002_S_1155_I995496" # MRI ID
        main(arg1, arg2, arg3)

    elif len(sys.argv) == 2:
        arg1 = sys.argv[1] # slice number
        arg2 = "002_S_1155" # Id Patien
        arg3 = "002_S_1155_I995496" # MRI ID
        main(arg1, arg2, arg3)
    
    elif len(sys.argv) == 3:
        arg1 = sys.argv[1] # slice number
        arg2 = sys.argv[2] # Id Patien
        arg3 = "002_S_1155_I995496" # MRI ID
        main(arg1, arg2, arg3)
    
    elif len(sys.argv) == 4:
        arg1 = sys.argv[1] # slice number
        arg2 = sys.argv[2] # Patien ID
        arg3 = sys.argv[3] # MRI ID
        main(arg1, arg2, arg3)
    
    else:
        print("Usage: python view_3D_tensor.py slice_number Id_patient MRI_Patient \nThe IDs are optional")
        sys.exit(1)
