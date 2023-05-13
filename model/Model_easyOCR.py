import os
import json
from PIL import Image
from easyocr import Reader
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2


class pre_train():
    def __init__(self ,languages ,  image_dir  , annotation_path):
        self.image_dir = image_dir
        self.annotation_path = annotation_path
        self.reader = Reader(languages)

        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        self.image_files = os.listdir(self.image_dir)


    
    
    
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('L')

        # Extract text boxes from annotations
        info_of_each_img = []
        img_name = os.path.basename(image_path)
        img_id = None
        for img in self.annotations["images"]:
            if img_name == img["file_name"]:
                img_id = img["id"]
                break
        for annotation in self.annotations["annotations"]:
            if annotation["image_id"] == img_id:
                bbox = annotation["bbox"]
                x0, y0, w, h = bbox
                x1, y1 = x0 + w, y0 + h
                text = annotation.get("attributes", {}).get("value")
                image_id = annotation["id"]
                info_of_each_img.append([image_id , text , [[x0, y0], [x1, y1]]])

        image = F.resize(image, (100, 32))
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)



        # Convert text to tensor
        text_list = [info[1] for info in info_of_each_img]
        text_tensor = self.reader.recognize(image_np, text_list)

        bbox_list = []
        for info in info_of_each_img:
            x0, y0, x1, y1 = info[2][0][0], info[2][0][1], info[2][1][0], info[2][1][1]
            bbox_list.append([y0, x0, y1, x1])  # convert to [y0, x0, y1, x1] format
        bbox_tensor = torch.tensor(bbox_list).float()

        # Convert image ID to tensor
        img_id_list = [info[0] for info in info_of_each_img]
        img_id_tensor = torch.tensor(img_id_list).long()

        return torch.tensor(image_np).float(), text_tensor, bbox_tensor, img_id_tensor

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def init_hidden(self, batch_size):
        return (torch.zeros(2, batch_size, self.hidden_size).to(device), 
                torch.zeros(2, batch_size, self.hidden_size).to(device))

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        try: # multi gpu needs this
            self.rnn.flatten_parameters()
        except: # quantization doesn't work with this 
            pass
        batch_size, T, _ = input.size()
        hidden = self.init_hidden(batch_size)
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class CRNN(nn.Module):

    def __init__(self, input_channel, num_classes, hidden_size):
        super(CRNN, self).__init__()
        self.FeatureExtractor = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.Prediction = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        visual_feature = self.FeatureExtractor(input)
        contextual_feature = self.SequenceModeling(visual_feature)
        prediction = self.Prediction(contextual_feature[:, -1, :])
        return prediction


class Model(nn.Module):

    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        self.cnn = CRNN(input_channel, output_channel, hidden_size)
        self.rnn = BidirectionalLSTM(output_channel, hidden_size, num_class)

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = self.rnn(conv)
        return output.permute(1, 0, 2)






    # def extract_text_box(self , image_path , annotations):
    #     info_of_each_img = []
    #     img_name = os.path.basename(image_path)
    #     img_id = None
    #     for img in annotations["images"]:
    #         if img_name == img["file_name"]:
    #             img_id = img["id"]
    #             break
    #     for annotation in annotations["annotations"]:
    #             if annotation["image_id"] == img_id:
    #                 bbox = annotation["bbox"]
    #                 x0, y0, w, h = bbox
    #                 x1, y1 = x0 + w, y0 + h
    #                 text = annotation.get("attributes", {}).get("value")
    #                 image_id = annotation["id"]
    #                 info_of_each_img.append([image_id , text , [[x0, y0], [x1, y1]]])
    #     return info_of_each_img
    # def crop_text_regions(self, image_path, results , image_id):
    #     img = Image.open(image_path)
    #     for res in results:
    #         text = res[1]
    #         bbox = res[0]
    #         conf = res[2]
    #         box = (bbox[0][0], bbox[0][1], bbox[2][0], bbox[3][1])
    #         cropped_img = img.crop(box)
    #         img_name = image_id
    #         cropped_img.save(f"./cropped image/{img_name}.png" , format="PNG")  
            
    # def pre_train(self):
    #     annotations , image_files = self.get_annotations_and_image_file()
    #     for img in image_files:
    #         if img.endswith('.png'):
    #             image_path = os.path.join(self.image_dir , img)
    #             info = self.extract_text_box(image_path, annotations)
    #             print(info)
    #             boxes = []
    #             image_id = []
    #             for i in info:
    #                 boxes.append(i[1])
    #                 image_id.append(i[0])
    #             detail = {'box': boxes}
    #             results = self.reader.readtext(image_path , detail=detail)
    #             for i in image_id:
    #               self.crop_text_regions(image_path, results , i)
                  
 

dataset = pre_train(languages=['en', 'ja'], image_dir='./business_card_dataset/images', annotation_path='./business_card_dataset/annotations/instances_default.json')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = Model(input_channel=3, output_channel=256, hidden_size=256, num_class=128)


for batch in dataloader:
    images, text, bbox, img_id = batch
    output = model(images, text)


# output + error : File "c:/Users/Fateme/Desktop/buisiness card/Model.py", line 222, in <module>
#     for batch in dataloader:
#   File "C:\Users\Fateme\AppData\Local\Programs\Python\Python38\lib\site-packages\torch\utils\data\dataloader.py", line 634, in __next__
#     data = self._next_data()
#   File "C:\Users\Fateme\AppData\Local\Programs\Python\Python38\lib\site-packages\torch\utils\data\dataloader.py", line 678, in _next_data
#     data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
#   File "C:\Users\Fateme\AppData\Local\Programs\Python\Python38\lib\site-packages\torch\utils\data\_utils\fetch.py", line 51, in fetch
#     data = [self.dataset[idx] for idx in possibly_batched_index]
#   File "C:\Users\Fateme\AppData\Local\Programs\Python\Python38\lib\site-packages\torch\utils\data\_utils\fetch.py", line 51, in <listcomp>
#     data = [self.dataset[idx] for idx in possibly_batched_index]
#   File "c:/Users/Fateme/Desktop/buisiness card/Model.py", line 61, in __getitem__
#     text_tensor = self.reader.recognize(image_np, text_list)
#   File "C:\Users\Fateme\AppData\Local\Programs\Python\Python38\lib\site-packages\easyocr\easyocr.py", line 379, in recognize
#     image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height = imgH)
#   File "C:\Users\Fateme\AppData\Local\Programs\Python\Python38\lib\site-packages\easyocr\utils.py", line 559, in get_image_list
#     x_min = max(0,box[0])
# TypeError: '>' not supported between instances of 'str' and 'int'
