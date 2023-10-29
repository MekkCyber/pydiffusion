import os
import random
import shutil
from pathlib import Path
# Paths
source_directory = Path('caltech101/101_ObjectCategories')
destination_directory = "images/"
num_images_to_select = 5

# Iterate through subdirectories in the source directory
for category_folder in os.listdir(source_directory):
    category_path = os.path.join(source_directory, category_folder)
    
    # Check if it's a directory
    if os.path.isdir(category_path):
        # List all files in the category folder
        images = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        if len(images) >= num_images_to_select:
            # Select 5 random images
            selected_images = random.sample(images, num_images_to_select)
            
            # Create destination folder for the category
            destination_category_path = os.path.join(destination_directory, category_folder)
            os.makedirs(destination_category_path, exist_ok=True)
            
            # Copy selected images to the destination folder
            for image in selected_images:
                source_image_path = os.path.join(category_path, image)
                destination_image_path = os.path.join(destination_category_path, image)
                shutil.copyfile(source_image_path, destination_image_path)
