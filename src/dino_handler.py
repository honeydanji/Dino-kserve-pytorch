from albumentations.pytorch.transforms import ToTensorV2
from ts.torch_handler.base_handler import BaseHandler
from dino_model import UnifiedClassifier
from PIL import Image
import torch, torch.hub
import albumentations as A
import numpy as np
import base64, os, cv2, io
import json

class DinoHandler(BaseHandler):
    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None
    def initialize(self, context):

        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Device 설정
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Model 가중치 파일 확인
        model_pt_path = 'C:/Personal/Dino/chpt/dino.pt'
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = UnifiedClassifier()
        self.model.load_state_dict(torch.load(model_pt_path, map_location=self.device))
        self.model.eval()

        self.initialized = True

    def preprocess(self, data):

        ## 데이터 전처리 코드 주석 처리함
        input_image = base64.b64decode(data["instances"][0]["data"])
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
        preprocessed_data = input_tensor

        return preprocessed_data

    def inference(self, data):
        # data : 전처리된 데이터
        # data 를 이용해서 실제 모델을 가져와 추론 진행 하는 코드 작성 해야 함.
        # 현재는 임시로 결과값 보냄.
        return self.model(data)

    def postprocess(self, data):
        # 추론 결과 후처리
        return json.dump(data)

    def handle(self, data, context):
        self.context = context

        if data=="none":
            return json.dumps({
                "test" : "a"
            })
        else:
            return Exception("에러발생")

        # try:
        #     if not self.initialized:
        #         self.initialize()
        #
        #     model_input = self.preprocess(data)
        #     model_output = self.inference(model_input)
        #     result = self.postprocess(model_output)
        #     return result
        #
        # except RuntimeError as e:
        #     raise Exception("에러 발생 하면 안되는데 ............." + str(e))
