import os
import shutil
from PIL import Image

def process_hpatch_dataset(root_dir):
    for subdir in os.listdir(root_dir):
        if subdir.startswith('i_') or subdir.startswith('v_'):
            subdir_path = os.path.join(root_dir, subdir)
            
            if os.path.isdir(subdir_path):
                ref_image_path = os.path.join(subdir_path, '1.ppm')
                trans_image_path = os.path.join(subdir_path, '2.ppm')
                
                if os.path.exists(ref_image_path) and os.path.exists(trans_image_path):
                    try:
                        ref_image = Image.open(ref_image_path)
                        trans_image = Image.open(trans_image_path)
                        
                        if ref_image.size != trans_image.size:
                            print(f"Removing directory: {subdir} (shape mismatch)")
                            shutil.rmtree(subdir_path)
                        else:
                            print(f"Keeping directory: {subdir}")
                    except Exception as e:
                        print(f"Error processing {subdir}: {str(e)}")
                else:
                    print(f"Skipping {subdir}: Missing required images")

# Usage
root_directory = "./hpatches-sequences-release"
process_hpatch_dataset(root_directory)