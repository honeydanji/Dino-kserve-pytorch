import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import os
import cv2
from PIL import Image
import io
import json
import base64
from ts.torch_handler.base_handler import BaseHandler

class Wall_Classifier(nn.Module):
    def __init__(self, num_classes, backbone, dino_backbones, backbone_name):
        super(Wall_Classifier, self).__init__()

        self.backbones = dino_backbones
        self.backbone = backbone
        self.backbone.eval()
        self.head = nn.Linear(self.backbones[backbone_name]['embedding_size'],num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = self.head(x)
        x = F.sigmoid(x)
        return x
    
class Tile_Classifier(nn.Module):
    def __init__(self, num_classes, backbone, dino_backbones, backbone_name):
        super(Tile_Classifier, self).__init__()

        self.backbones = dino_backbones
        self.backbone = backbone
        self.backbone.eval()
        self.head = nn.Linear(self.backbones[backbone_name]['embedding_size'],num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = self.head(x)
        x = F.sigmoid(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self, num_classes, backbone, dino_backbones):
        super(Classifier, self).__init__()

        self.backbones = dino_backbones
        self.backbone = torch.hub.load('facebookresearch/dinov2', self.backbones[backbone]['name'])
        self.backbone.eval()
        self.head = nn.Linear(self.backbones[backbone]['embedding_size'],num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = self.head(x)
        return x
    
class UnifiedClassifier(nn.Module):
    def __init__(self):
        super(UnifiedClassifier, self).__init__()
        self.backbones = {
                                            'dinov2_s':{
                                                'name':'dinov2_vits14_reg',
                                                'embedding_size':384,
                                                'patch_size':14
                                            },
                                            'dinov2_b':{
                                                'name':'dinov2_vitb14_reg',
                                                'embedding_size':768,
                                                'patch_size':14
                                            },
                                            'dinov2_l':{
                                                'name':'dinov2_vitl14_reg',
                                                'embedding_size':1024,
                                                'patch_size':14
                                            },
                                            'dinov2_g':{
                                                'name':'dinov2_vitg14_reg',
                                                'embedding_size':1536,
                                                'patch_size':14
                                            },
                                        }
        self.backbone_name = 'dinov2_b'
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # 새 모델 로드 매커니즘을 통합합니다.
        self.load_models()
    
    def load_models(self):
        # Backbone 설정
        self.backbone = torch.hub.load('facebookresearch/dinov2', self.backbones[self.backbone_name]['name']).to(self.device)
        self.backbone.eval()
        
        # Wall Classifier 설정
        self.wall_model = Wall_Classifier(8, self.backbone, self.backbones, self.backbone_name).to(self.device)
        self.wall_model.eval()
        
        # Tile Classifier 설정
        self.tile_model = Tile_Classifier(4, self.backbone, self.backbones, self.backbone_name).to(self.device)
        self.tile_model.eval()
        
        # Main Classifier 설정
        self.model = Classifier(3, 'dinov2_b', self.backbones)
        self.model.eval()
    
    def forward(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            backbone_feature = self.backbone(input_tensor)
            first_cls_val, first_cls_pred = torch.max(F.softmax(self.model.head(backbone_feature), dim=1), dim=1)
            
            if first_cls_pred == 0:
                return {"prediction": [first_cls_val.tolist(), first_cls_pred.tolist(), None]}
            elif first_cls_pred == 1:  # tile
                tile_pred = torch.sigmoid(self.tile_model.head(backbone_feature)).tolist()
                return {"prediction": [first_cls_val.tolist(), first_cls_pred.tolist(), tile_pred]}
            elif first_cls_pred == 2:  # wall
                wall_pred = torch.sigmoid(self.wall_model.head(backbone_feature)).tolist()
                return {"prediction": [first_cls_val.tolist(), first_cls_pred.tolist(), wall_pred]}


class DinoHandler(BaseHandler):
    """
    A custom model handler implementation.
    """
    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """
        #  load the model
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # # Read model serialize/pt file
        # serialized_file = self.manifest['model']['serializedFile']
        # model_pt_path = os.path.join(model_dir, serialized_file)

        model_pt_path = '../chpt/dino.pt'

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = UnifiedClassifier()
        self.model.load_state_dict(torch.load(model_pt_path, map_location=self.device))
        self.model.eval()

        self.initialized = True
    
    def input_fn(self, request_body):
        # input_image = Image.open(io.BytesIO(request_body["instances"][0]["image"]["b64"]))
        input_image = base64.b64decode(request_body["instances"][0]["image"]["b64"])
        # print(input_image)
        input_image = Image.open(io.BytesIO(input_image))
        input_image = np.array(input_image)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        processing = A.Compose([
                A.Resize(520, 520),
                A.CenterCrop(518, 518),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=(0.8, 0.8), contrast_limit=(0,0), p=1.0),
                ToTensorV2()
                ])
        
        input_tensor = processing(image=input_image)['image'].unsqueeze(0)
        return input_tensor
    
    
    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        data = self.input_fn(data)
        pred = json.dumps(self.model(data))
        return pred