import os
import csv
import matplotlib.pyplot as plt
import networkx as nx
from mmdet.apis import init_detector, inference_detector
from mmcv import Config
from mmdet.apis import DetInferencer

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
def infer_images_in_folder(inferencer, src_dir, out_dir, G):
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
                    
                    for label in labels:
                        if label not in G:
                            G.add_node(label)
                        for other_label in labels:
                            if label != other_label:
                                if G.has_edge(label, other_label):
                                    G[label][other_label]['weight'] += 1
                                else:
                                    G.add_edge(label, other_label, weight=1)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# 可视化共现网络
def visualize_network(G):
    pos = nx.spring_layout(G)  # 改善布局
    plt.figure(figsize=(8, 8))  # 设置图形的大小
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='k', node_size=500, font_size=10, font_weight='bold')
    plt.show()

# 保存共现网络信息
def export_to_csv(edges, output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['node1', 'node2', 'weight'])
        for edge in edges:
            node1, node2, weight = edge[0], edge[1], edge[2]['weight']
            writer.writerow([node1, node2, weight])

            
if __name__ == "__main__":
    # 初始化检测器
    model = setup_detector(config_file, checkpoint_file, device)
    # 遍历文件夹并进行推理
    infer_images_in_folder(inferencer, src_dir, out_dir, G)
    # 输出共现网络节点
    print(G.nodes())
    # 可视化共现网络
    visualize_network(G)
    # 将共现网络的边信息输出到 CSV 文件中
    export_to_csv(list(G.edges(data=True)), output_filename)