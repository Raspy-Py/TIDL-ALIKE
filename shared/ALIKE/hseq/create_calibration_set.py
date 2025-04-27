import os
import shutil
from tqdm import tqdm

set_name = "hpatch_v"

src_dir = "hpatches-sequences-release"
dst_dir = f"../../assets/data/{set_name}"

# take only vieport variant
folders = [folder for folder in os.listdir(src_dir) if folder[0] == 'v']
print(f"folders found: {len(folders)}")

os.makedirs(dst_dir, exist_ok=True)

data_file = []

with open(os.path.join(dst_dir, f'../{set_name}.csv'), mode="w") as f: 
    data_file.append("image_folder_path,filename")
    for idx, folder in tqdm(enumerate(folders, start=1), desc="Extracting files: "):
        shutil.copy(
            os.path.join(src_dir, folder, "1.ppm"), 
            os.path.join(dst_dir, f"{idx}.ppm")
        )
        data_file.append(f"{dst_dir.split('/')[-1]},{idx}.ppm")
    f.write("\n".join(data_file))