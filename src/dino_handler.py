from albumentations.pytorch.transforms import ToTensorV2
from ts.torch_handler.base_handler import BaseHandler
from dino_model import UnifiedClassifier
from PIL import Image
import torch, torch.hub
import albumentations as A
import numpy as np
import base64, os, cv2, io
import json
import logging


class DinoHandler(BaseHandler):

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None
        self.logger = logging.getLogger(__file__)

    def initialize(self, context):
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("initialize start")
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        serialized_file = self.manifest['model']['serializedFile']

        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = UnifiedClassifier()
        self.model.load_state_dict(torch.load(model_pt_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.initialized = True

    def preprocess(self, data):
        self.logger.debug("preprocess start")
        data = data[0]['body'].decode('utf-8')
        data = json.loads(data)

        input_image = base64.b64decode(data["instances"][0]["data"])
        input_image = Image.open(io.BytesIO(input_image))
        input_image = np.array(input_image)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        processing = A.Compose([
            A.Resize(520, 520),
            A.CenterCrop(518, 518),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0,
                        always_apply=False, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(0.8, 0.8), contrast_limit=(0, 0), p=1.0),
            ToTensorV2()
        ])

        input_tensor = processing(image=input_image)['image'].unsqueeze(0)
        self.logger.debug("preprocess done")
        return input_tensor

    def inference(self, model_input):
        model_input = model_input.to(self.device)
        model_output = self.model(model_input)
        return model_output

    def postprocess(self, inference_output):
        postprocess_output = json.dumps(inference_output)
        return [postprocess_output]

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        model_output = self.postprocess(model_output)

        return model_output
