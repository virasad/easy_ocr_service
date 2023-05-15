import json
import os
import cv2
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import traceback


annotation_path ="./business_card_dataset/annotations/instances_default.json"
image_dir = "./business_card_dataset/images"

with open(annotation_path , 'r' , encoding='utf-8') as f:
        annotations = json.load(f)
        image_files = os.listdir(image_dir)

cropped_img = []
text = []

for i in image_files:
        if i.endswith('.png'):
                image_path = os.path.join(image_dir , i)
                img_name = os.path.basename(image_path)
                image = Image.open(image_path)
                if image is None: 
                    print(f"Error reading image: {image_path}")
                    continue
                img_id = None
                for img in annotations["images"]:
                    if img_name == img["file_name"]:
                        img_id = img["id"]
                        break
                for annotation in annotations["annotations"]:
                        if annotation["image_id"] == img_id:
                            bbox = annotation["bbox"]
                            x1 = bbox[0]
                            y1 = bbox[1]
                            x2 = bbox[0] + bbox[2]
                            y2 = bbox[1]
                            x3 = bbox[0] + bbox[2]
                            y3 = bbox[1] + bbox[3]
                            x4 = bbox[0]
                            y4 = bbox[1] + bbox[3]
                            points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                            crop_img = image.crop((min([p[0] for p in points]), min([p[1] for p in points]), max([p[0] for p in points]), max([p[1] for p in points])))

                            crop_img = cv2.resize(np.array(crop_img), (640, 480))
                            crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                            crop_img_blur = cv2.medianBlur(crop_img_gray, 5)
                            

                            crop_img_rgb = np.moveaxis(crop_img_blur, -1, 0)




                            cropped_img.append(crop_img_rgb)
                            txt = annotation.get("attributes", {}).get("value")
                            text.append(txt)
                           

x_train , x_test, y_train , y_test = train_test_split(cropped_img , text , test_size=0.2 , random_state=42)

x_train = torch.tensor(np.array(x_train)).float()
print(x_train.shape)
x_test = torch.tensor(np.array(x_test)).float()
print(x_test.shape)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_train = torch.tensor(y_train, dtype=torch.long)
print(y_train.shape)

y_test = label_encoder.fit_transform(y_test)
y_test = torch.tensor(y_test, dtype=torch.long)


train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
       
        try: 
            self.rnn.flatten_parameters()
        except: 
            pass
        recurrent, _ = self.rnn(input)  
        output = self.linear(recurrent)  
        return output

class VGG_FeatureExtractor(nn.Module):

    def __init__(self, input_channel, output_channel=256):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))

    def forward(self, input):
        print(input.shape)
        return self.ConvNet(input)

class Model(nn.Module):

    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)


    def forward(self, input, text):
        print(input.shape)
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        contextual_feature = self.SequenceModeling(visual_feature)

        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction


model = Model(input_channel=3, output_channel=256, hidden_size=256, num_class=128)

batch_size = 32
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for batch in train_dataloader:
    input, text = batch
    output = model(input, text)

                                                        


                
        
