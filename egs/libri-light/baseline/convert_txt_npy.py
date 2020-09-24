import sys
import os
import numpy as np
#this_name=sys.argv[0]
#var=sys.argv[1]
#print('argv[0] is ' + this_name + '; argv[1] is ' + var + '\n')

if len(sys.argv) > 1:
    npy_txt_path=sys.argv[1] # it should contain files ./name.npy.txt
else:
    print("should provide source directory name\n")
    exit()
#npy_txt_path_nosplash=npy_txt_path.split("/")
if npy_txt_path[-1] == '/':
    npy_path=npy_txt_path[:-1] + '_npy'
else:
    npy_path=npy_txt_path + '_npy'
# then make a output directory
print("source directory is:" + npy_txt_path)
#npy_path=npy_txt_path + "_npy"
print("making output directory: " + npy_path)
if not os.path.exists(npy_path):
    os.makedirs(npy_path)
npy_txt_files= os.listdir(npy_txt_path)
num_files = len(npy_txt_files)
print("Total file number:" + str(num_files))
for this_file in npy_txt_files:
    if this_file.endswith(".txt"):
        f = open(npy_txt_path + "/" + this_file);
        feat_mat = np.loadtxt(f)
        output_this_file = this_file[:-4] # this_file= xxx.npy.txt	 np.save(npy_path + "/" + output_this_file, feat_mat)
        np.save(npy_path + "/" + output_this_file, feat_mat)

print(npy_txt_path + ": done converting to npy format...")
