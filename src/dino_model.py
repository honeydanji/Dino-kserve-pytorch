import torch
import torch.nn as nn
import torch.nn.functional as F

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

