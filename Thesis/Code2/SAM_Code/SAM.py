import os
#config = os.environ["CUDA_VISIBLE_DEVICE"] = "0"
import numpy as np
import torch
import cv2
from pathlib import Path
import nibabel as nib
import time
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

sam_checkpoint = "/home/rraciti/Tesi/Codice/Code_SAM/sam_vit_h_4b8939.pth"
model_type = "vit_h"


sam = sam_model_registry[model_type](checkpoint= sam_checkpoint)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam.to(device=device)


if torch.cuda.is_available():
    print("The model is loaded in: ", sam.device)
else:
    print("No GPU available. The model is loaded in CPU.")


mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=50,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)


def Chek(path):
  val = 0
  for img in path.iterdir():
    n_slice = str(img).split("_")[-1].split("e")[1]
    if int(n_slice) > val:
      val = int(n_slice)

  return val

directorys = Path("/home/rraciti/Tesi/Dataset/Image_NiFTY_Longitudinal")
for folder in directorys.iterdir():

  #print(f"Folder -> {folder}")
  for patient in folder.iterdir():
    start_time_patient = time.time() #start patient time measurement

    #print(f"Patient -> {patient}")
    patient_id = str(patient).split("/")[-2]

    for image in patient.iterdir():
     #print(f"Image -> {image}")
      image_ID = str(image).split('/')[-1].split('.')[0]
      #print(f"Image ID -> {image_ID}")

      masksPath = Path("/home/rraciti/Tesi/Results/SAM_result/Masks") / patient_id / image_ID

      #print(f"MaskPath -> {masksPath}")


      try:
        masksPath.mkdir()
        starting_point = 0
      except FileExistsError:
        print(f"The folder {masksPath} is already present")
        starting_point = Chek(masksPath) 
      

      nifti_path = str(image)
      nifti_image = nib.load(nifti_path)
      data = nifti_image.get_fdata()

      for slice in range(256):
        if slice >= starting_point:
          start_time_slice = time.time() #start slice time measurement

          corrent_slice = f"{masksPath}/{image_ID}_slice{slice}"
          masksSlicePath = Path(corrent_slice)

          #print(f"masksSlicePath -> {masksSlicePath}")

          try:
            masksSlicePath.mkdir()
          except FileExistsError:
            print(f"The folder {masksSlicePath} is already present")
            number_of_element = len(list(masksSlicePath.iterdir()))
            if  number_of_element != 0: continue
          


          
          img = data[:, :, slice][:, :, np.newaxis]

          img = np.repeat(img, 3, axis=2)

          img = 255 - (255 * (img.max() - img) / (img.max() - img.min())).astype(np.uint8)



          print(f"Calculation of slice masks n°{slice}")
          masks = mask_generator.generate(img)
          sort_mask = sorted(masks, key=lambda x: x['area'], reverse=True)
          filtered_masks = [mask for mask in sort_mask if mask['area'] >= 50]

          print(f"Reduction of masks from {len(masks)} to {len(filtered_masks)}")

          number_slice = str(masksSlicePath).split('/')[-1]
          for i in range(len(filtered_masks)):

            img_np = (filtered_masks[i]['segmentation'])
            img_tt = torch.from_numpy(img_np)
            img_tt = img_tt[None, : ]

            img = img_tt.long().type(torch.uint8)

            mask_name = f"{number_slice}_{i}.png"
            print(mask_name)

            tensor = img.permute(1, 2, 0)
            array = tensor.numpy()

            cv2.imwrite(str(masksSlicePath / mask_name), array)
          
          end_time_slice = time.time()
          execution_time_slice = end_time_slice - start_time_slice
          print(f"Runtime single Slice: {execution_time_slice}")

  end_time_patient = time.time()
  execution_time_patient = end_time_patient - start_time_patient
  print(f"Runtime patient: {execution_time_patient}")

torch.cuda.empty_cache()



