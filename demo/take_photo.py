import os
import cv2
from utils import ImageLoader
from datetime import datetime



device = "/dev/video2"
save_folder = "calibration"


if __name__ == "__main__":
    image_loader = ImageLoader(device)

    image = image_loader.read()

    if image is not None:
        print(f"Image shape is ({image.shape})")
        current_time = datetime.now().strftime("%H-%M-%S")
        path = os.path.join(save_folder, f"camera-capture-{current_time}.png")
        print("Saving image to ", path)
        if not cv2.imwrite(path, image):
            print("Failed to save image!")
    else:
        print("Failed to capture image")
