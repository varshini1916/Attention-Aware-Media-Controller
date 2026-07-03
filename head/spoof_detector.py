import os
import cv2
import numpy as np

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name


class SpoofDetector:

    def __init__(self):
        self.model_test = AntiSpoofPredict(0)
        self.image_cropper = CropImage()
        self.model_dir = "./resources/anti_spoof_models"

    def detect(self, frame):

        image_bbox = self.model_test.get_bbox(frame)

        prediction = np.zeros((1, 3))

        for model_name in os.listdir(self.model_dir):

            h_input, w_input, model_type, scale = parse_model_name(model_name)

            param = {
                "org_img": frame,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }

            img = self.image_cropper.crop(**param)

            prediction += self.model_test.predict(
                img,
                os.path.join(self.model_dir, model_name)
            )

        label = np.argmax(prediction)

        confidence = prediction[0][label]

        if label == 1:
            return f"LIVE USER {confidence:.2f}", (0,255,0)
        else:
            return f"SPOOF {confidence:.2f}", (0,0,255)