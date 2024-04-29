import os
import numpy as np
import piexif
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load and prepare the image
def load_and_prepare_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# Load the model
model = MobileNetV2(weights='imagenet')

# Function to get keywords from an image
def get_keywords_from_image(image):
    image_prepared = load_and_prepare_image(image)
    preds = model.predict(image_prepared)
    predictions = decode_predictions(preds, top=10)  # Get top-10 predictions
    keywords = [pred[1].replace('_', ' ') for pred in predictions[0]]
    return keywords

# Generate a basic description from keywords
def generate_description_from_keywords(keywords):
    return 'Image containing ' + ', '.join(keywords[:5]) + '.'

# Function to process all images in a folder, update metadata, and rename files
def process_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing image: {image_path}")
            keywords = get_keywords_from_image(image_path)
            description = generate_description_from_keywords(keywords)
            update_image_metadata(image_path, keywords, description)
            rename_image(image_path, description)

# Update the image metadata
def update_image_metadata(image_path, keywords, description):
    keywords_string = ';'.join(keywords)
    exif_dict = piexif.load(image_path)
    exif_dict['0th'][piexif.ImageIFD.XPKeywords] = keywords_string.encode('utf-16le')
    exif_dict['0th'][piexif.ImageIFD.ImageDescription] = description.encode('utf-8')
    exif_bytes = piexif.dump(exif_dict)
    
    with Image.open(image_path) as img:
        img.save(image_path, "jpeg", exif=exif_bytes)
        print(f"Updated metadata for {image_path}")

# Rename the image based on description
def rename_image(image_path, description):
    folder_path, extension = os.path.splitext(image_path)
    new_filename = description.replace(' ', '_').replace(',', '').replace('.', '') + extension
    new_filename = os.path.join(os.path.dirname(image_path), new_filename)
    os.rename(image_path, new_filename)
    print(f"Renamed image to {new_filename}")

# Path to your folder containing images
folder_path = r"D:\Image_to_text\images"  # Change this to the path of your folder
process_images_in_folder(folder_path)
