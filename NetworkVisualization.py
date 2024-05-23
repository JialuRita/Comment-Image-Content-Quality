import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# 编号对应的物体标签映射
LABELS = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic_light',
    10: 'fire_hydrant',
    11: 'stop_sign',
    12: 'parking_meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports_ball',
    33: 'kite',
    34: 'baseball_bat',
    35: 'baseball_glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis_racket',
    39: 'bottle',
    40: 'wine_glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot_dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted_plant',
    59: 'bed',
    60: 'dining_table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell_phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy_bear',
    78: 'hair_drier',
    79: 'toothbrush'
}

# 读取CSV文件并创建网络图
def read_csv_to_network(csv_file, label_map, weight_threshold):
    df = pd.read_csv(csv_file)
    G = nx.Graph()
    for index, row in df.iterrows():
        node1, node2, weight = row['node1'], row['node2'], row['weight']
        node1 = label_map.get(node1, f"Unknown({node1})")
        node2 = label_map.get(node2, f"Unknown({node2})")
        if weight > weight_threshold:
            G.add_edge(node1, node2, weight=weight)
    return G

# 网络可视化
def visualize_network(G, weight_threshold):
    pos = nx.shell_layout(G)  # 自适应布局
    plt.figure(figsize=(15, 15))  # 设置图形的大小
    nx.draw(
        G, pos, 
        with_labels=True, 
        node_color='lightblue', 
        edge_color='lightgrey', 
        node_size=700, 
        font_size=12, 
        font_color='black', 
        font_weight='bold', 
        linewidths=1.5
    )
    # 显示边的权重
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(
        G, pos, 
        edge_labels=edge_labels, 
        font_color='red', 
        font_size=10, 
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
    )
    # 标题和边框
    plt.title(f"Network Visualization (Weight Threshold: {weight_threshold})", size=20, color='darkgreen')
    plt.axis('off')  # 关闭坐标轴
    # 保存图像
    plt.savefig(f"./Images/img_{weight_threshold}.png", bbox_inches='tight', pad_inches=0.1)
    # plt.show()

if __name__ == "__main__":
    csv_file_path = './Output/object_co_occurrence_network.csv'
    # 改变共现阈值
    for weight_threshold in [10, 20, 30, 50, 100, 200, 300]:
        G = read_csv_to_network(csv_file_path, LABELS, weight_threshold)  #生成新的共现网络（删除低于阈值的边）
        print(weight_threshold)
        print(f"Number of edges: {len(G.edges)}")
        print(f"Number of nodes: {len(G.nodes)}")
        # 可视化网络图
        visualize_network(G, weight_threshold)
