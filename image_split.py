from PIL import Image
import os

def split_and_save_images(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            # Load the image
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path)
            
            # Get image dimensions
            width, height = image.size
            
            # Calculate the width of each vertical section
            split_width = width // 4
            
            # Define coordinates for each vertical split
            vertical_splits = [
                (i * split_width, 0, (i + 1) * split_width, height) for i in range(4)
            ]
            
            # Crop and save each vertical section
            for i, coords in enumerate(vertical_splits, start=1):
                crop = image.crop(coords)
                split_filename = f"{os.path.splitext(filename)[0]}_split_{i}.tiff"
                crop.save(os.path.join(output_dir, split_filename))
                
    print(f"Vertical splits saved to '{output_dir}'")

# Example usage
input_dir = "input_images"  # Directory containing the images to split
output_dir = "output_vertical_splits"  # Directory to save cropped images
split_and_save_images(input_dir, output_dir)
