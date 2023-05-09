import os
from easyocr import Reader
from sklearn.model_selection import train_test_split
import json
from PIL import Image
import re

class easyOCR:
    def __init__(self , languages = ['en' , 'ja'] , image_dir = "./business_card_dataset/images" , annotation_path ="./business_card_dataset/annotations/instances_default.json"):
          self.languages = languages
          self.image_dir = image_dir
          self.annotation_path = annotation_path
          self.reader  = Reader(languages)

    # Open and read the annotaion
    def get_annotations_and_image_file(self):
        with open(self.annotation_path , 'r' , encoding='utf-8') as f:
            annotations = json.load(f)
        image_files = os.listdir(self.image_dir)
        return annotations , image_files
    
    def extract_text_box(self , image_path , annotations):
        texts = []
        boxes = []
        img_name = os.path.basename(image_path)
        img_id = None
        for img in annotations["images"]:
            if img_name == img["file_name"]:
                img_id = img["id"]
                break
        for annotation in annotations["annotations"]:
                if annotation["image_id"] == img_id:
                    bbox = annotation["bbox"]
                    x0, y0, w, h = bbox
                    x1, y1 = x0 + w, y0 + h
                    boxes.append([[x0, y0], [x1, y1]])
                    text = annotation.get("attributes", {}).get("value")
                    if text:
                        texts.append(text)
        return texts , boxes

    def crop_text_regions(self, image_path, results):
        img = Image.open(image_path)
        for res in results:
            text = res[1]
            bbox = res[0]
            conf = res[2]
            box = (bbox[0][0], bbox[0][1], bbox[2][0], bbox[3][1])
            cropped_img = img.crop(box)
            img_name = "".join(c if c.isalnum() else "_" for c in text)
            cropped_img.save(f"{img_name}.png" , format="PNG")  
            
    def train(self):
        annotations , image_files = self.get_annotations_and_image_file()
        train_files , test_files = train_test_split(image_files, test_size=0.1 , random_state=42)
        for img in train_files:
            if img.endswith('.png'):
                image_path = os.path.join(self.image_dir , img)
                texts, boxes = self.extract_text_box(image_path, annotations)
                detail = {'box': boxes}
                results = self.reader.readtext(image_path , detail=detail)
                self.crop_text_regions(image_path, results)

    def test(self):
        annotations , image_files = self.get_annotations_and_image_file()
        train_files , test_files = train_test_split(image_files, test_size=0.1 , random_state=42)
        for img in test_files:
            if img.endswith('.png'):
                image_path = os.path.join(self.image_dir , img)
                texts, boxes = self.extract_text_box(image_path, annotations)
                detail = {'box': boxes}
                results = self.reader.readtext(image_path , detail=detail)
                self.crop_text_regions(image_path, results)
                for res in results:
                    text = res[1]
                    bbox = res[0]
                    conf = res[2]
                    print(f"System output : {text} - Confidence: {conf}")
                    
                       


model = easyOCR()
model.train()
model.test()
