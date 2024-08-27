from .base_predictor import Predictor
import torch
import os
import cv2
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from .yolox.exp import get_exp
from .yolox.utils import postprocess
from .yolox.data.data_augment import ValTransform

class MegviiPredictor(Predictor):
    def __init__(self, framework, models_params):
        super().__init__("megvii", framework)
        self.models_params = models_params
        self.exps = [get_exp(model_param['exp_file']) for model_param in models_params]
        self.device = "cpu"
        
    def load_one_model(self, model_param):
        exp = get_exp(model_param['exp_file'])
        model = exp.get_model()
        model.eval()
        ckpt = torch.load(model_param['weight'], map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if self.device == "gpu":
            model.cuda()
        self.models.append((model, model_param))

    def predict_one_model(self, model, image, model_param):
        img_info = {"id": 0}
        img_info["file_name"] = os.path.basename(image)
        img = cv2.imread(image)
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(model_param['img_size'] / img.shape[0], model_param['img_size'] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = ValTransform(legacy=False)(img, None, (model_param['img_size'], model_param['img_size']))
        img = torch.from_numpy(img).unsqueeze(0).float()
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            outputs = model(img)
            outputs = postprocess(
                outputs, self.exps[0].num_classes, model_param['conf'], self.exps[0].nmsthre, class_agnostic=True
            )
        
        boxes, scores, labels = [], [], []
        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy()
            for output in outputs:
                x1, y1, x2, y2 = output[:4] / img_info["ratio"]
                score, cls_id = output[4] * output[5], output[6]
                boxes.append(self.normalize_box([x1, y1, x2, y2], width, height))
                scores.append(score)
                labels.append(cls_id + 1)
        
        return boxes, scores, labels

    def predict(self):
        boxes_list, scores_list, labels_list = [], [], []
        for model, model_param in self.models:
            model_boxes, model_scores, model_labels = [], [], []
            for image in self.images:
                b, s, l = self.predict_one_model(model, image, model_param)
                model_boxes.append(b)
                model_scores.append(s)
                model_labels.append(l)
            boxes_list.append(model_boxes)
            scores_list.append(model_scores)
            labels_list.append(model_labels)
        return boxes_list, scores_list, labels_list

    def load(self, models_params, images_path):
        self.images = self.load_images(images_path)
        for model_param in models_params:
            print(f"Loading model from weight: {model_param['weight']}")
            self.load_one_model(model_param)
            # print(f"Current models: {self.models}")