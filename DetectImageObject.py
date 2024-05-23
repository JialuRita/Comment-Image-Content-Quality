import os
import json
import cv2
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

setup_logger()

# 加载detectron2
def setup_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置模型阈值
    cfg.MODEL.DEVICE = "cpu"  # 使用CPU
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor

# 从final_sample.json中加载图片路径
def load_image_paths(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data.get("pic_path", {})

# 单张图片的物体检测
def detect_objects(predictor, image_path):
    img = cv2.imread(image_path)
    outputs = predictor(img)
    return outputs

# 物体检测
def process_images(predictor, image_paths):
    detected_objects = {}
    for key, path in image_paths.items():
        if os.path.exists(path):
            outputs = detect_objects(predictor, path)  # 单张图片的物体检测
            detected_objects[key] = outputs["instances"].pred_classes.cpu().numpy().tolist()
    return detected_objects

# 保存结果
def save_detected_objects(detected_objects, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detected_objects, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 设置预测器
    predictor = setup_predictor()
    # 加载图片路径
    image_paths = load_image_paths('./NewData/final_sample.json')
    # 处理图片，进行物体检测
    detected_objects = process_images(predictor, image_paths)
    # 保存检测结果
    save_detected_objects(detected_objects, './NewData/detected_objects.json')
