import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# Function to split the image into four vertical parts
def split_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    split_width = width // 4
    vertical_splits = [
        (i * split_width, 0, (i + 1) * split_width, height) for i in range(4)
    ]
    return [image.crop(coords) for coords in vertical_splits]

# Function to annotate splits and combine them into one final image
def annotate_and_combine_splits(model, image_parts, base_filename, output_dir, undefined_dir):
    # Get dimensions of each split
    part_width, part_height = image_parts[0].size
    total_width = part_width * len(image_parts)
    total_height = part_height

    # Create a blank image to combine the splits
    combined_image = Image.new('RGB', (total_width, total_height))
    combined_draw = ImageDraw.Draw(combined_image)
    font = ImageFont.truetype("arial.ttf", 60)  # Font for annotation

    # Process each part
    for i, part in enumerate(image_parts):
        # Resize part to 224x224 and preprocess
        part_resized = part.resize((224, 224))
        img_array = img_to_array(part_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        # Predict
        prediction = model.predict(img_array)
        good_confidence = prediction[0][1]  # Assuming index 1 is 'good'
        bad_confidence = prediction[0][0]   # Assuming index 0 is 'bad'

        # Determine label and color based on threshold
        if max(good_confidence, bad_confidence) < 0.52:
            label = "Un-identified"
            color = "orange"
            # Save undefined split
            undefined_path = os.path.join(undefined_dir, f"{base_filename}_part_{i+1}_undefined.jpg")
            part.save(undefined_path)
        elif good_confidence > bad_confidence:
            label = "Good"
            color = "green"
        else:
            label = "Bad"
            color = "red"

        # Paste the part into the combined image
        combined_image.paste(part, (i * part_width, 0))

        # Draw bounding box around the part
        top_left = (i * part_width, 0)
        bottom_right = ((i + 1) * part_width - 1, part_height - 1)
        combined_draw.rectangle([top_left, bottom_right], outline="blue", width=3)

        # Add annotation (label with confidence scores) in the center of each part
        label_with_confidence = f"{label} ({good_confidence:.2f}/{bad_confidence:.2f})"
        text_bbox = combined_draw.textbbox((0, 0), label_with_confidence, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = (
            i * part_width + (part_width - text_width) // 2,
            (part_height - text_height) // 2
        )
        combined_draw.text(text_position, label_with_confidence, fill=color, font=font)

    # Save the final combined image
    final_image_path = os.path.join(output_dir, f"{base_filename}_final_output.jpg")
    combined_image.save(final_image_path)

    return final_image_path

# Load the trained model
model = load_model('efficientnetb7_model_finetuned(2).h5')

# Directory containing the images to predict
input_dir = 'input_images'
output_dir = 'output_predictions'
undefined_dir = 'undefined_images'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(undefined_dir, exist_ok=True)

# Process each image
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        image_path = os.path.join(input_dir, filename)

        # Split the image into four vertical parts
        image_parts = split_image(image_path)

        # Annotate and combine splits into one final image
        base_filename = os.path.splitext(filename)[0]
        final_output_path = annotate_and_combine_splits(model, image_parts, base_filename, output_dir, undefined_dir)

        print(f"Final output for {filename} saved at {final_output_path}.")
