import numpy as np
import os
import itertools

def npz_to_csv(filename:str, folder:str):
    print("Converting file "+ filename)
    file = np.load(folder+filename)
    new_folder = "data/csv/"+dist_type+"/"
    new_filename = new_folder+filename[:filename.index(".")]+".csv"
    
        
    header = "|"
    arrays = []
    for key, value in file.items():
        if len(value.shape) == 0:
            arrays.append([value])
            header +=  key + "|"
        elif len(value.shape) == 2:
            for i, arr in enumerate(value.T):
                arrays.append(arr)
                header += key +"_"+str(i)+"|"
        else:
            arrays.append(value)
            header +=  key + "|"

    in_list = list(map(list, itertools.zip_longest(*arrays, fillvalue=" "*len(str(arrays[0])))))
    np.savetxt(new_filename, in_list, header=header, fmt = "%s", delimiter=",")
    file.close()


if __name__ == "__main__":
    dist_type = "dippeak"
    dist_type = "mixed"

    folder = "data/"+dist_type+"/"
    filenames = os.listdir("./"+folder)

    for filename in filenames:
        npz_to_csv(filename, folder)
