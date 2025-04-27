import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

def plot_detected_points(rgb_images, scoremaps, keypoints, output_file='detected_points.png'):
    N = len(rgb_images)
    
    fig, axes = plt.subplots(N, 3, figsize=(10, N * 3))

    # Handle the case where N == 1
    if N == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = ['Source image', 'Scores Map', 'Detected Keypoints']
    for ax, col in zip(axes[0], column_titles):
        ax.set_title(col)

    for i in range(N):
        image = np.transpose(rgb_images[i], (0, 2, 3, 1))
        axes[i, 0].imshow(image.squeeze(0))
        axes[i, 0].axis('off')

        scoremaps_img = scoremaps[i][1].squeeze(0).squeeze(0)
        axes[i, 1].imshow(scoremaps_img, cmap='viridis')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(scoremaps_img, cmap='viridis')
        axes[i, 2].scatter(keypoints[i][:, 0], keypoints[i][:, 1], c='r', s=1)
        axes[i, 2].axis("off")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def create_gif(image_sequence, filename, duration=0.1):
    imageio.mimsave(filename, image_sequence, duration=duration)