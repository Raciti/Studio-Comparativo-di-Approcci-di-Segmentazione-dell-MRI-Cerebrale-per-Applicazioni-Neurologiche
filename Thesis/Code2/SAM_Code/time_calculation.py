slice_time = 0
n_sclies = 0
patient_time = 0
n_patients  = 0

sliceTime = []
patientTime = []
patients = []

with open('./2D_to_3D.log', 'r') as file:
    for line in file:

        if "Slice Dir:" in line:
            patient = line.split(" ")[-1]
            patient = patient[:-1] #Used to remove \n
            n_patients += 1
            patients.append(patient)

        if "slice:" in line:
            times = line.split(' ')[-1]
            n_sclies += 1
            slice_time += float(times)

        if "patient:" in line:
            timep = line.split(' ')[-1]
            patient_time += float(timep)
            patientTime.append(float(timep))
            sliceTime.append(slice_time)
            n_sclies = 0
            slice_time = 0


with open('/home/rraciti/Tesi/Results/SAM_result/Runtime/runtime.txt', 'w') as file:
    # Scrive del testo nel file
    file.write("3D image creation\n")
    file.write(f"Patients: {patients} \nPatients time:{patientTime}\nTotal time slice:{sliceTime}\nAverage slice time:{sum(sliceTime)/len(sliceTime)}\nAverage patients time:{sum(patientTime)/n_patients}")



slice_time = 0
n_sclies = 0
n_patients  = 0

sliceTime = []
sliceTime_average = []
patientTime = []

with open('./nohup.out', 'r') as file:
    for line in file:
        
        if "Slice" in line:
            times = float(line.split(':')[-1])
            slice_time += times
            n_sclies += 1


        if "patient:" in line:
            sliceTime.append(slice_time)
            sliceTime_average.append(slice_time/n_sclies)
            slice_time = 0
            n_sclies = 0
            timep = float(line.split(':')[-1])
            patientTime.append(timep)
            n_patients += 1



with open('/home/rraciti/Tesi/Results/SAM_result/Runtime/runtime.txt', 'a') as file:
    # Scrive del testo nel file
    file.write("\n\nMasks SAM generator\n")
    file.write(f"Patients Time: {patientTime} \nAverage patients time: {sum(patientTime)/n_patients}\nSlice Time: {sliceTime}\nAverage Slice time: {sliceTime_average}")
