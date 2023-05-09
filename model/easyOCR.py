import os
from easyocr import Reader
from sklearn.model_selection import train_test_split
import json
from PIL import Image

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
        for annotation in annotations["annotations"]:
                if annotation.get('filename') == os.path.basename(image_path) and annotation["language"] in self.languages:
                    boxes.append(annotation["bbox"])
                    texts.append(annotation["text"])
                return texts , boxes
        
    def train(self):
        annotations , image_files = self.get_annotations_and_image_file()
        train_files , test_files = train_test_split(image_files, test_size=0.1 , random_state=42)
        for img in train_files:
            if img.endswith('.png'):
                image_path = os.path.join(self.image_dir , img)
                texts, boxes = self.extract_text_box(image_path, annotations)
                detail = {'box': boxes}
                results = self.reader.readtext(image_path , detail=detail)

    def test(self):
        annotations , image_files = self.get_annotations_and_image_file()
        train_files , test_files = train_test_split(image_files, test_size=0.1 , random_state=42)
        for img in test_files:
            if img.endswith('.png'):
                image_path = os.path.join(self.image_dir , img)
                texts, boxes = self.extract_text_box(image_path, annotations)
                detail = {'box': boxes}
                results = self.reader.readtext(image_path , detail=detail)
                for res in results:
                    text = res[1]
                    bbox = res[0]
                    conf = res[2]
                    print(f"System output : {text} - Confidence: {conf}")
                    if all(isinstance(coord, int) for coord in bbox):
                        bbox = tuple(int(coord) for coord in bbox)
                        try:

                            region = Image.open(image_path).crop(bbox)
                            region.save(f"{text}.png")
                        except ValueError:
                           pass

model = easyOCR()
model.train()
model.test()
