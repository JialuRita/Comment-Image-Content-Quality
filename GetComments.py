import os
import json

# 图像物体检测与网络生成
def infer_images_in_folder(src_dir,processed_image_paths):
    count = 0
    for subdir, dirs, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                file_path = os.path.join(subdir, file)
                count += 1
                print(f"{count}: Processing {file_path}")
                try:
                    processed_image_paths.append(file_path)  # 保存处理过的图片路径
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# 保存处理过的图片路径到本地文件
def save_processed_image_paths(processed_image_paths, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for path in processed_image_paths:
            new_path = path.replace('E:/rita/complex_network/week12/20201016ImgData/', 'I:\\')
            new_new_path = new_path.replace('/', '\\').replace('\\', '\\\\')  # 替换路径中的斜杠
            print(new_new_path)
            f.write(f"{new_new_path}\n")

# 提取图片对应的评论相关属性
def extract_matching_attributes(path_file, second_json_file, output_file):
    # 加载文件路径
    paths = []
    with open(path_file, 'r', encoding='utf-8') as f:
        paths = f.read().splitlines()
    # 加载第二个JSON文件
    second_data = load_json(second_json_file)
    # 提取匹配的属性
    matching_attributes = []
    pic_path_dict = second_data.get('pic_path', {})
    content_dict = second_data.get('content', {})
    for path in paths:
        # 替换路径中的斜杠以确保匹配
        normalized_path = path.replace('/', '\\').replace('\\\\', '\\')
        for key, value in pic_path_dict.items():
            if value == normalized_path:
                matching_attributes.append({
                    'pic_path': value,
                    'comment': content_dict.get(key, "No comment found")
                })
                print('find')
                break  # 一旦找到匹配项，就跳出循环
    # 将匹配的属性保存到新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matching_attributes, f, ensure_ascii=False, indent=4)
    return matching_attributes

# 加载JSON文件
def load_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 保存JSON文件
def save_json(data, json_file):
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def change_path(path):
    # 读取JSON文件
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    new_data = []
    # 替换路径
    for item in data:
        if 'pic_path' in item:
            print(item['pic_path'])
            item['pic_path'] = item['pic_path'].replace('I:\\', 'E:\\rita\\complex_network\\week12\\20201016ImgData\\')
            print(item['pic_path'])
        new_data.append(item)
    print(new_data)
    save_json(new_data, './NewData/new_pic_comment.json')
    


if __name__=='__main__':
    src_dir = "E:/rita/complex_network/week12/20201016ImgData/jd_comment_picture/pic_data_high_pix_downsample/4"
    output_file = "./NewData/pic_path.txt"
    # 遍历文件夹获取图片路径
    processed_image_paths = []  # 用于保存处理过的图片路径
    infer_images_in_folder(src_dir, processed_image_paths)
    save_processed_image_paths(processed_image_paths, output_file)

    # 提取图片对应的评论属性
    path_file = output_file  # 路径
    second_json_file = './Data for Effect of User-Generated Image on Review Helpfulness Perspectives from Object Detection/tabular data/target_comment_seed2021.json'  # 第二个JSON文件路径
    output_json_file = './NewData/pic_comment.json'  # 输出路径
    extracted_data = extract_matching_attributes(path_file, second_json_file, output_file)
    save_json(extracted_data, output_json_file)
    change_path('./NewData/pic_comment.json')