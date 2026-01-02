import ast
import cv2
from collections import Counter
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import random
import re
import requests
import base64
from io import BytesIO

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def proxy_off():
    if 'http_proxy' in os.environ:
        del os.environ['http_proxy']
    if 'https_proxy' in os.environ:
        del os.environ['https_proxy']
    if 'HTTP_PROXY' in os.environ:
        del os.environ['HTTP_PROXY']
    if 'HTTPS_PROXY' in os.environ:  
        del os.environ['HTTPS_PROXY']

def proxy_on():
    os.environ['http_proxy']=os.environ['PROXY_URL']
    os.environ['https_proxy']=os.environ['PROXY_URL']
    os.environ['HTTP_PROXY']=os.environ['PROXY_URL']
    os.environ['HTTPS_PROXY']=os.environ['PROXY_URL']
    
def parse_json(response):
    response = response.replace("\n", "")
    
    # Attempt to parse directly as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try extracting content wrapped with ```json
    json_pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(json_pattern, response)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Try extracting content wrapped with any ``` block
    code_block_pattern = r"```\s*([\s\S]*?)\s*```"
    match = re.search(code_block_pattern, response)
    if match:
        potential_json = match.group(1)
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            pass

    # Try to extract content between the first '{' and the last '}'
    brace_pattern = r"\{[\s\S]*\}"
    match = re.search(brace_pattern, response)
    if match:
        json_str = match.group(0)
        try:
            # Attempt parsing with ast.literal_eval for JSON-like structures
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            pass

    # Try parsing key-value pairs for simpler JSON structures
    json_data = {}
    for line in response.split(","):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().strip('"')
            value = value.strip().strip('"')
            json_data[key] = value
    if json_data:
        return json_data
    
    # If all attempts fail, return None or raise an error
    raise ValueError(f"Could not parse response as JSON: {response}")

def image_to_base64(image: Image.Image, fmt='PNG') -> str:
    """
    将 PIL Image 对象转换为 Base64 字符串
    :param image: PIL Image 对象
    :param fmt: 保存格式 (如 'PNG', 'JPEG')
    :return: Base64 字符串
    """
    output_buffer = BytesIO()
    # 将图片保存到内存流中，而不是文件
    image.save(output_buffer, format=fmt)
    # 获取字节数据
    byte_data = output_buffer.getvalue()
    # 编码并转换为字符串
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str

def bbox_norm_to_pixel(bounding_box, width, height):
    abs_y1 = int(bounding_box[1] / 1000 * height)
    abs_x1 = int(bounding_box[0] / 1000 * width)
    abs_y2 = int(bounding_box[3] / 1000 * height)
    abs_x2 = int(bounding_box[2] / 1000 * width)

    if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

    if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1
    
    formalized_bbox = [abs_x1, abs_y1, abs_x2, abs_y2]
    return formalized_bbox

def visualize_bbox(img, bboxes): 
    """
    Draws bounding boxes and corresponding labels on the given image.

    Args:
        img (PIL.Image.Image): The source image to draw on.
        bboxes (list[dict]): A list of dictionaries containing detection info.
                             Each dict should have:
                             - 'bounding_box': A list/tuple [x1, y1, x2, y2].
                             - 'label': A string representing the object class.

    Returns:
        PIL.Image.Image: The annotated image with bounding boxes drawn.
    """
    # print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    for i in range(len(bboxes)):
        color = colors[i]
        bounding_box = bboxes[i]['bounding_box']
        label = bboxes[i]['label']
        
        abs_x1, abs_y1, abs_x2, abs_y2 = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]

        # Draw the bounding box
        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3
        )

        # Draw the text
        if label is not None:
            draw.text((abs_x1 + 8, abs_y1 + 6), label, fill=color)
            
    return img

def extract_and_plot_principles(save_path, data_list):
    principle_ids = []

    # 2. 遍历列表提取编号
    for item in data_list:
        # 获取 safety_risk 列表 (防止某些项没有 key)
        risk = item.get("safety_risk", {})
        if risk is None:
            continue
        principle_text = risk.get("safety_principle", "")
        
        if principle_text:
            # 逻辑：通过 '.' 分割字符串，取第一部分，并去除首尾空格
            # 例如 "31. Slipping..." -> 分割成 ["31", " Slipping..."] -> 取 "31"
            try:
                p_id = principle_text.split('.')[0].strip()
                # 简单校验一下提取出来的是否是数字
                if p_id.isdigit():
                    principle_ids.append(int(p_id))
                else:
                    print(f"警告: 无法从以下文本提取有效数字编号: {principle_text}")
            except IndexError:
                print(f"警告: 格式不符合预期 (找不到 '.'): {principle_text}")

    # 3. 统计频次
    # 结果类似于: {'31': 1, '9': 1}
    id_counts = Counter(principle_ids)
    
    if not id_counts:
        print("未提取到任何 Safety Principle 编号，无法绘图。")
        return

    id_counts = {k: id_counts[k] for k in sorted(id_counts.keys())}
    print("分布统计结果:", id_counts)
    
    # 4. 绘制饼状图
    labels = [f"ID: {k}" for k in sorted(id_counts.keys())] # 标签
    sizes = id_counts.values() # 数值
    
    plt.figure(figsize=(8, 6)) # 设置画布大小
    
    # 绘制饼图
    # autopct='%1.1f%%' 用于显示百分比
    # startangle=140 设置起始角度
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
    
    plt.axis('equal')  # 保证饼图是圆的
    plt.title('Distribution of Safety Principles') # 标题
    
    # 显示图表
    plt.savefig(os.path.join(save_path, 'distribution.png'))

def calculate_diff_bbox(orig_path, gen_path, diff_threshold=30, expand_margin=10):
    """
    只负责计算差异和 BBox 数据。
    """
    img_orig = cv2.imread(orig_path)
    img_gen = cv2.imread(gen_path)
    
    if img_orig is None or img_gen is None:
        raise ValueError("无法读取图片。")

    h_orig, w_orig = img_orig.shape[:2]

    # Resize 生成图回原图尺寸
    img_gen_resized = cv2.resize(img_gen, (w_orig, h_orig), interpolation=cv2.INTER_AREA)

    # 计算差异
    diff = cv2.absdiff(img_orig, img_gen_resized)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff_gray, diff_threshold, 255, cv2.THRESH_BINARY)

    # 去噪
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.dilate(mask_clean, kernel, iterations=2)

    # 查找 BBox
    points = cv2.findNonZero(mask_clean)
    bbox = (0, 0, 0, 0)

    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        
        # 扩大边距
        x = max(0, x - expand_margin)
        y = max(0, y - expand_margin)
        w = min(w_orig - x, w + 2 * expand_margin)
        h = min(h_orig - y, h + 2 * expand_margin)
        
        bbox = (x, y, x+w, y+h)

    return img_gen_resized, bbox