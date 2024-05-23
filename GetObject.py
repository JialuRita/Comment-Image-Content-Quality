import os
import json
import networkx as nx
import os
import csv
import matplotlib.pyplot as plt
import networkx as nx
from mmdet.apis import init_detector, inference_detector
from mmcv import Config
from mmdet.apis import DetInferencer
import re
import jieba
import synonyms


# 设置全局变量
config_file = 'rtmdet_tiny_8xb32-300e_coco.py'  # 配置文件路径
checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'  # 检查点文件路径
device = 'cpu'
inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device=device)
src_dir = "E:/rita/complex_network/week12/20201016ImgData/jd_comment_picture/pic_data_high_pix_downsample/4"
out_dir = r'outputs'
output_filename = './Output/object_co_occurrence_network.csv'

# 初始化无向图
G = nx.Graph()

# 构建mmdetection检测器
def setup_detector(config_file, checkpoint_file, device='cpu'):
    model = init_detector(config_file, checkpoint_file, device=device)
    return model

# 图像物体检测与网络生成
def infer_images_in_folder(inferencer, src_dir, out_dir, detected_objects):
    count = 0
    for subdir, dirs, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                file_path = os.path.join(subdir, file)
                count += 1
                print(f"{count}: Processing {file_path}")
                try:
                    result = inferencer(file_path, print_result=True, out_dir=out_dir)
                    res = result['predictions'][0]
                    labels = res['labels'][:15]
                    scores = res['scores'][:15]
                    labels = [label for label, score in zip(labels, scores) if score > 0.3]

                    detected_objects.append({
                        "pic_path": file_path,
                        "object": labels
                    })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# 保存检测到的物体信息到本地文件
def save_detected_objects(detected_objects, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detected_objects, f, ensure_ascii=False, indent=4)


# 定义一个函数来提取文本中的物品名称及其同义词
def extract_objects_with_synonyms(text, synonyms_instance, k=4):
    # 使用正则表达式匹配文本中的物品名称（这里需要一个匹配物品名称的正则表达式）
    words = jieba.lcut(text)
    # 提取包含物品名称的词语
    objects = [word for word in words if re.match(r'[\u4e00-\u9fa5]+', word)]
    # 创建一个集合来存储物品及其同义词
    objects_set = set()
    # 遍历物品名称，查找并添加同义词
    for obj in objects:
        # 获取物品的同义词列表
        synonyms_list = synonyms_instance.get(obj, k=k)
        # 将物品名称及其同义词添加到集合中
        objects_set.update([syn for syn in synonyms_list if syn is not None])
    return objects_set


if __name__=='__main__':
    synonyms_instance = synonyms.Synonyms()    
    # 读取JSON文件
    path_comment = './NewData/pic_comment.json'
    with open(path_comment, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        path = item['pic_path']
        comment = item['comment']
        object = item['object']
        # 计算 Otext
        Otext = extract_objects_with_synonyms(comment, synonyms_instance)
        #从图片中检测到的物品集合 Oimg
        Oimg = object
        # 计算 Jaccard 相似度
        jaccard_similarity = len(Otext.intersection(Oimg)) / len(Otext.union(Oimg))
        print(jaccard_similarity)
