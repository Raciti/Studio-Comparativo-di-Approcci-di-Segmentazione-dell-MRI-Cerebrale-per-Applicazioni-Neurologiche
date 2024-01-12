from pathlib import Path
import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt
import time

def getName(path):
  name = str(path).split('/')[-1]
  return name

def getIndex(name):
  idx = name.split('e')[1]
  return int(idx)

def getTensor(path):
  tensor = read_image(str(path))
  tensor = tensor.permute(1,2,0)
  return tensor

def show(tensor):
  plt.figure(figsize=(6,6))
  plt.imshow(tensor)
  plt.axis('off')
  plt.show()
  plt.savefig('/home/rraciti/Tesi/Codice/Code_SAM/show.png')

def findBackground(path):
  idx = str(path).split(".")[-2].split("_")[2]
  if idx == "0": return True
  return False

def createPath(path, patient):
    return f"{path}/{getName(patient)}"

def makeDirectory(path):
   try:
      Path(path).mkdir()
   except FileExistsError:
      print(f"The directory {path} is already present")

def name_3D_image(path_3D_image, patient, sliceDir):
   return f"{createPath(path_3D_image, getName(patient))}/{getName(patient)}_{getName(sliceDir)}.pt"

path_3D_image = "/home/rraciti/Tesi/Results/SAM_result/3D_Image"

Masks = Path("/home/rraciti/Tesi/Results/SAM_result/Masks")
for patient in Masks.iterdir():
    time_patient_start = time.time()
    for sliceDir in patient.iterdir():
      print(f"Slice Dir: {getName(sliceDir)}")
      print(len(list(sliceDir.iterdir())))
      img3d_list = [1] * len(list(sliceDir.iterdir())) #list of overlapping masks
      print(len(img3d_list))
      for slices in sliceDir.iterdir():        
        print(f"Slices: {getName(slices)}")
        time_slice_start = time.time()
        tensorSlice = torch.zeros((208, 240, 1), dtype= torch.uint8)
        for slice in slices.iterdir():
            if findBackground(getName(slice)) == False:  
                slice = getTensor(slice)
                tensorSlice = tensorSlice + slice
    
        img3d_list[getIndex(getName(slices))] = tensorSlice
        time_slice_end = time.time()
        time_slice = time_slice_end - time_slice_start
        print(f"Computation time slice: {time_slice}")


      img3d = torch.stack(img3d_list, dim=0) #final 3d image

      makeDirectory(createPath(path_3D_image, getName(patient)))
      
      torch.save(img3d, name_3D_image(path_3D_image, patient, sliceDir))

      time_patient_end = time.time()
      time_patient = time_patient_end - time_patient_start
      print(f"Computation time patient: {time_patient}")


