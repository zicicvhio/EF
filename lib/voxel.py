import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

image_folder = 'path_to_your_image_folder'  
image_files = sorted(os.listdir(image_folder))  

if len(image_files) < 2:
    print("error")
    exit()

threshold = 0.05  

img_path = os.path.join(image_folder, image_files[0])
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
height, width = image.shape

time_window_size = len(image_files)  
voxel_grid = np.zeros((height, width, 3, time_window_size))  

previous_image = None
time = 0  

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)

    current_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if current_image is None:
        print(f"can't read：{img_path}")
        continue

    current_image = current_image.astype(np.float32) / 255.0

    if previous_image is None:
        previous_image = current_image
        continue

    delta_image = current_image - previous_image

    event_positions = np.where(np.abs(delta_image) > threshold)

    x_positions = event_positions[1]
    y_positions = event_positions[0]
    event_strength = delta_image[y_positions, x_positions]

    event_timestamp = time * np.ones_like(x_positions)  
    voxel_grid[y_positions, x_positions, 0, time] += 1 
    voxel_grid[y_positions, x_positions, 1, time] += event_strength 
    voxel_grid[y_positions, x_positions, 2, time] += event_timestamp  
 
    time += 1  
 previous_image = current_image
lambda_values = [0.1, 0.3, 0.5, 0.8, 1.0]  # 不同的lambda值

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for lambda_weight in lambda_values:
   
    image_resized = cv2.resize(current_image, (width, height))

    fused_voxel_grid = voxel_grid.copy()
    fused_voxel_grid[:, :, 0, :] = fused_voxel_grid[:, :, 0, :] * lambda_weight + (
                image_resized[:, :, np.newaxis] * (1 - lambda_weight))
    
    event_coords = np.where(fused_voxel_grid[:, :, 0, :] > 0)
    x, y, z = event_coords
    
    ax.scatter(x, y, z, c=z, cmap='jet', marker='o', s=1, label=f"lambda={lambda_weight}")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time')
ax.set_title("Event Distribution with Different lambda Values")

# not use loc="best"
ax.legend(loc='upper right')  

plt.show()
