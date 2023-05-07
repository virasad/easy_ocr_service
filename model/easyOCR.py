import os
from easyocr import Reader
from sklearn.model_selection import train_test_split
import json
from PIL import Image, ImageDraw , ImageFont



# Direction of images and nnotation
image_dir = "./business_card_dataset/images"
annotation_path ="./business_card_dataset/annotations/instances_default.json"

# Open and read the annotaion
with open(annotation_path , 'r' , encoding='utf-8') as f:
    annotations = json.load(f)

# Set languages that gonna use in the project
languages = ['en' , 'ja']

# Split dataset into training and testing sets
image_files = os.listdir(image_dir)
train_files, test_files = train_test_split(image_files, test_size=0.1, random_state=42)

# Create easyOCR reader classs
reader = Reader(languages)

# Loop thtough each image file
for img in train_files:

    # Check the extension(.PNG)
    if img.endswith('.png'):

        # Get the full path to the image and annotation file
        image_path = os.path.join(image_dir , img)

        image = os.path.basename(image_path)

        # Extract text and bounding box from annotation
        texts = []
        boxes = []
        for annotation in annotations["annotations"]:
                if annotation.get('filename') == image and annotation["language"] in languages:
                    boxes.append(annotation["bbox"])
                    texts.append(annotation["text"])

        # Create a dictionary with the bounding boxes
        detail = {'box': boxes}

        # Train the model using image and annotation
        results = reader.readtext(image_path , detail=detail)
        



# Test the model using the testing set
for img in test_files:
    if img.endswith('.png'):
        image_path = os.path.join(image_dir , img)

        image = os.path.basename(image_path)
        texts = []
        boxes = []
        for annotation in annotations["annotations"]:
                if annotation.get('filename') == image and annotation["language"] in languages:
                    boxes.append(annotation["bbox"])
                    texts.append(annotation["text"])
        detail = {'box': boxes}
        results = reader.readtext(image_path , detail=detail)
        for res in results:
             text = res[1]
             bbox = res[0]
             conf = res[2]
             print(f"System output : {text} - Confidence: {conf}")

             # Crop text region
             if all(isinstance(coord, int) for coord in bbox):
                 bbox = tuple(int(coord) for coord in bbox)
                 try:
                     region = Image.open(image_path).crop(bbox)
                     region.save(f"{text}.png")
                 except ValueError:
                     pass
         
