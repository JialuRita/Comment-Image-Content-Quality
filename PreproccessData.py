import json
from random import sample
from PIL import Image
import os

# 加载JSON文件
def load_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 加载ID对应的所有图片
def try_load_all_images_for_id(image_path, image_count):
    all_paths = [os.path.join(os.path.dirname(image_path), f"{i}.jpg") for i in range(image_count)]
    for path in all_paths:
        new_path = path.replace('I:\\', 'E:\\rita\\complex_network\\week12\\20201016ImgData\\')  # 修改为本地的图片保存路径
        try:
            Image.open(new_path)  # 尝试打开图片
        except (IOError, OSError):
            return False  # 任意一张图片加载失败，返回False
    return True  # 所有图片都成功加载，返回True

# 抽取满足条件的数据
def extract_data_with_existing_images(data, num):
    valid_entries = {'pic_path': {}, 'imageCount': {}}
    all_pic_paths = data.get('pic_path', {})
    image_counts = data.get('imageCount', {})
    # 筛选出存在的图片路径
    for id_, path in all_pic_paths.items():
        image_count = image_counts.get(id_, 1)
        if try_load_all_images_for_id(path, image_count):
            valid_entries['pic_path'][id_] = path.replace('I:\\', 'E:\\rita\\complex_network\\week12\\20201016ImgData\\')  # 修改为本地的图片保存路径
            valid_entries['imageCount'][id_] = image_count
    # 随机抽取请求的样本数：5000
    if len(valid_entries['pic_path']) > num:
        sampled_ids = sample(list(valid_entries['pic_path'].keys()), num)
        valid_entries['pic_path'] = {id_: valid_entries['pic_path'][id_] for id_ in sampled_ids}
        valid_entries['imageCount'] = {id_: valid_entries['imageCount'][id_] for id_ in sampled_ids}
    return valid_entries

# 提取图片对应的评论相关属性
def extract_matching_attributes(first_json_file, second_json_file):
    # 加载第一个JSON文件（抽取后）
    first_data = load_json(first_json_file)
    pic_path_ids = set(first_data.get('pic_path', {}).keys())
    image_count_ids = set(first_data.get('imageCount', {}).keys())
    # 确保ID一致的
    assert pic_path_ids == image_count_ids, "ID sets in pic_path and imageCount do not match"
    # 加载第二个JSON文件
    second_data = load_json(second_json_file)
    # 提取匹配的属性
    matching_attributes = {}
    for key, value in second_data.items():
        if key != 'pic_path' and key != 'imageCount':
            matching_attributes[key] = {id_: second_data[key][id_] for id_ in value if id_ in pic_path_ids}
    # 将pic_path和imageCount复制到新的JSON文件
    matching_attributes['pic_path'] = first_data['pic_path']
    matching_attributes['imageCount'] = first_data['imageCount']
    return matching_attributes

# 保存JSON文件
def save_json(data, json_file):
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        

if __name__ == '__main__':
    # 数据的随机抽样
    json_file = './Data for Effect of User-Generated Image on Review Helpfulness Perspectives from Object Detection/tabular data/target_comment_seed2021.json'  # JSON文件路径
    data = load_json(json_file)
    extracted_data = extract_data_with_existing_images(data, 5000)  #抽取5000条数据
    save_json(extracted_data, './NewData/samples_step1_dispose.json')
    # 提取图片对应的评论属性
    first_json_file = './NewData/samples_step1_dispose.json'  # 第一个JSON文件路径
    second_json_file = './Data for Effect of User-Generated Image on Review Helpfulness Perspectives from Object Detection/tabular data/target_comment_seed2021.json'  # 第二个JSON文件路径
    output_json_file = './NewData/final_sample.json'  # 输出路径
    extracted_data = extract_matching_attributes(first_json_file, second_json_file)
    save_json(extracted_data, output_json_file)