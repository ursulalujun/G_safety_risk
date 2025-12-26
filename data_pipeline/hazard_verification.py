import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import torch
import os
import openai
from PIL import Image, ImageDraw
from utils import visualize_bbox, image_to_base64, parse_json, bbox_norm_to_pixel
from tqdm import tqdm

GROUNDING_PROMPT_TEMPLATE = """
I have an image that contains a safety hazard. 

Context:
- Hazard Type: {hazard_type}
- Safety Principle: {safety_principle}{instruction_context}
- Edition Plan: {edit_desc}

Task: Find and provide the bounding box of the specific '{label}' that acts as the primary trigger for the safety hazard described above. 

Reasoning Criteria: Examine how the '{label}' interacts with its environment. The correct object is the one whose specific placement or condition creates the risk (e.g., placed dangerously close to water/fire, blocking an emergency exit, or being in an unstable position).

Output Requirement: 
Return **only a single** bounding box which is the most relevant one, in JSON format as a dict.
```json
{{
    "bbox": list # [xmin, ymin, xmax, ymax] (normalized to 0-1000). If no item matching the label is detected, output None
}}
```
"""

STATE_CHECK_TEMPLATE="""
You are a **Safety Hazard Auditor**. Your task is to verify if a synthesized image accurately reflects a specific {hazard_type} safety hazard description.

We have edited an image to create a safety hazard based on the following plan:
- Safety Principle (to be violated): {safety_principle}{instruction}
- Edit Description: "{edit_desc}"
- Hazard-related Objects: {hazard_objects}

Please meticulously analyze the **Hazard-related Objects** in the edited image across three dimensions. You must "look" at the details before you "score".

### 1. State Analysis (Visual Attributes)
Check if the object's physical state matches the `{edit_desc}`.
- **Surface & Texture**: Are there specific details like stains, water droplets, cracks, or texture changes?
- **Active State**: If the description implies activity (e.g., "boiling", "red-hot", "open"), are there visual cues like steam, glowing elements, or an ajar door?

### 2. Spatial Analysis (Position & Geometry)
Check the spatial relationship of hazard-related object aligns with the `{edit_desc}` and creates a valid **trigger condition** for the hazard.
- **Proximity & Interaction**: Is the hazard-related object placed closely enough to a hazard source to realistically trigger an accident (e.g., combustible material within ignition range of a stove)?
- **Stability & Physics**: Is the hazard-related object positioned in a physically unstable state (e.g., a fragile vase perched on the very edge of a table, ready to fall)?

### 3. Hazard Validity (Risk Logic)
Combine the state and spatial analysis.
- Does this specific combination of state and position violate the `{safety_principle}`?
- Is the risk unambiguous to a human observer?

Based on your analysis, output a single JSON object with the following structure:

```json
{{
  "analysis_trace": {{
    "state_observation": "Describe what you see regarding the object's state (e.g., 'The plate surface is visibly dirty with dark stains').",
    "spatial_observation": "Describe the position (e.g., 'The cloth is placed 5cm from the lit gas burner').",
    "risk_conclusion": "Explain why this constitutes a hazard (or why it fails)."
  }},
  "scores": {{
    "state_fidelity": 0-10,  // How well does the object's look match the description?
    "spatial_precision": 0-10, // How accurately does the object's spatial position match the description?
    "hazard_severity": 0-10  // How obvious and severe is the safety violation?
  }},
  "final_answer": "ACCEPTED" | "REJECTED", // PASS if all scores > 7 and risk is clear
  "refinement_suggestion": "If FAIL, provide a specific instruction to fix it (e.g., 'Move the cup closer to the edge' or 'Add steam to the water')."
}}
```
"""

class HazardVerifier:
    def __init__(self, detector_model):
        key = os.getenv("ANNOTATION_API_KEY")
        url = os.getenv("ANNOTATION_API_URL")
        self.client = openai.OpenAI(api_key=key, base_url=url)
        self.detector = detector_model

    def detect(self, image, label, risk_info, hazard_type):
        """
        直接调用 Qwen 进行 Grounding 获取最相关的 bbox
        """
        base64_image = image_to_base64(image)

        edit_desc = risk_info["edition_plan"]
        safety_principle = risk_info["safety_principle"]
        if hazard_type == "environmental":
            ins_context = f""
        else:
            instruction = risk_info["instruction"]
            ins_context = f"\n- Action Instruction: {instruction}\n"
        
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                prompt = GROUNDING_PROMPT_TEMPLATE.format(
                    hazard_type=hazard_type, 
                    safety_principle=safety_principle, 
                    edit_desc=edit_desc, 
                    instruction_context=ins_context, 
                    label=label
                )

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                response = self.client.chat.completions.create(
                    model=self.detector,
                    messages=messages,
                    temperature=0.0, # Grounding 任务建议低采样
                ).choices[0].message.content

                # 处理 Thinking 模型的输出
                if "</think>" in response:
                    response = response.split('</think>')[-1].strip()
                
                result = parse_json(response)
                width, height = image.size
                if result['bbox'] is not None:
                    return bbox_norm_to_pixel(result['bbox'], width, height)
                else:
                    return None
            except Exception as e:
                print(f"⚠️ [Attempt {attempt}/{max_retries}] | Error: {e}")
        return None


    def verify_object(self, image_path, pil_image, objects_to_detect, risk, hazard_type):
        """
        循环对象列表，直接使用 Qwen 获取坐标
        """
        all_detected_boxes = [] # 用于最后的可视化展示

        for role, obj_label in objects_to_detect:
            # 直接调用 Qwen Grounding
            final_box = self.detect(pil_image, obj_label, risk, hazard_type)
            
            if final_box is None:
                error_info = f"REJECTED: [Missing Hazard-Related Area] {role}: {obj_label}"
                risk["hazard_check"] = error_info
                return # 如果有一个没找着，直接返回
            
            # 保存坐标
            if role == "hazard_area":
                risk["bbox_annotation"][obj_label] = final_box
            else:
                risk["bbox_annotation"][role][obj_label] = final_box
            
            all_detected_boxes.append({"label": obj_label, "bounding_box": final_box})
        
        risk["hazard_check"] = "ACCEPTED"
            
        # --- 可视化与保存 ---
        save_path = image_path.replace('edit_image', 'annotate_image')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        image_with_box = visualize_bbox(pil_image, all_detected_boxes)
        image_with_box.save(save_path)

    def verify_state(self, image, risk, hazard_type):
        base64_image = image_to_base64(image)
        
        instruction = risk.get("instruction", "")
        edit_desc = risk.get("edit_description", "")
        safety_principle = risk.get("safety_principle", {})
        hazard_objects = risk.get("Hazard_related_area", {})
        if instruction != "":
            ins_context = f"\n- Action Instruction: {instruction}\n"
        else:
            ins_context = ""
        
        prompt = STATE_CHECK_TEMPLATE.format(hazard_type=hazard_type, hazard_objects=hazard_objects, safety_principle=safety_principle, edit_desc=edit_desc, instruction=ins_context)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.detector,
            messages=messages,
        ).choices[0].message.content

        check_result = parse_json(response)

        if "accepted" in check_result["final_answer"].lower(): 
            risk["hazard_check"] = "ACCEPTED"
        else:
            refinement_suggestion = check_result["refinement_suggestion"]
            risk["hazard_check"] = f"REJECTED: {refinement_suggestion}"

def process_single_item(item, verifier, hazard_type):
    """
    处理单个数据项的逻辑函数
    """
    risk = item["safety_risk"]
    
    # 过滤条件
    if risk is None or 'rejected' in risk['fidelity_check'].lower():
        return item, "Skipped (Fidelity)"

    image_path = risk["edit_image_path"]
    if not image_path or not os.path.exists(image_path):
        return item, f"Skipped (File not found: {image_path})"
        
    # 处理图像
    pil_image = Image.open(image_path).convert("RGB")
    objects_to_detect = []
    hazard_objs = risk["hazard_related_area"]

    # 构建检测列表
    if hazard_type.lower() == "environmental":
        risk["bbox_annotation"] = {}
        # 假设 hazard_objs 在 environmental 下是列表
        for obj_name in hazard_objs:
            objects_to_detect.append(("hazard_area", obj_name))
    else:
        target_objs = hazard_objs.get("target_object", [])
        constraint_objs = hazard_objs.get("constraint_object", [])
        
        risk["bbox_annotation"] = {
            "target_object": {},
            "constraint_object": {}
        }
        if target_objs:
            for name in target_objs:
                objects_to_detect.append(("target_object", name))
        if constraint_objs:
            for name in constraint_objs:
                objects_to_detect.append(("constraint_object", name))

    # --- 第一步：标注 BBox ---
    verifier.verify_object(image_path, pil_image, objects_to_detect, risk, hazard_type)

    # --- 第二步：检查空间关系 ---
    if risk.get("hazard_check") == "ACCEPTED":
        verifier.verify_state(pil_image, risk, hazard_type)
    
    return item, "Success"

def verify_hazard(meta_file_path, save_path, detector_name, hazard_type, max_workers):
    if "qwen" in detector_name.lower():
        if 'http_proxy' in os.environ:
            del os.environ['http_proxy']
        if 'https_proxy' in os.environ:
            del os.environ['https_proxy']
        if 'HTTP_PROXY' in os.environ:
            del os.environ['HTTP_PROXY']
        if 'HTTPS_PROXY' in os.environ:  
            del os.environ['HTTPS_PROXY']
    else:
        proxy = 'http://luxiaoya:kwMUZpsjfkRdN6rANEJp45sBoXK9gP1uLzQbwgerNbixbWFj3iOQMjTynOq8@proxy.h.pjlab.org.cn:23128'
        os.environ['http_proxy']=proxy
        os.environ['https_proxy']=proxy
        os.environ['HTTP_PROXY']=proxy
        os.environ['HTTPS_PROXY']=proxy
        
    # 初始化验证器
    # 注意：如果 HazardVerifier 内部涉及 GPU 模型，请确保它是线程安全的，
    # 或者将 max_workers 设置为较小的值以防显存溢出。
    verifier = HazardVerifier(detector_name)

    with open(meta_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    failed_items = []

    import ipdb; ipdb.set_trace()
    process_single_item(data[0], verifier, hazard_type)
    
    print(f"🚀 Starting parallel processing with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_item = {
            executor.submit(process_single_item, item, verifier, hazard_type): i 
            for i, item in enumerate(data)
        }

        with tqdm(total=len(data), desc="🖼️ 验证图像中") as pbar:
            for future in as_completed(future_to_item):
                idx = future_to_item[future]
                try:
                    # 获取结果（item 会被原位修改，因为它是 mutable 的）
                    _, status = future.result()
                except Exception as e:
                    error_info = {"index": idx, "error": str(e)}
                    failed_items.append(error_info)
                    print(f"Error: {error_info}")
                finally:
                    pbar.update(1)

    # 打印错误摘要
    if failed_items:
        print(f"⚠️ Totally {len(failed_items)} failure case.")

    # 保存结果
    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 处理完成，结果已保存至: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hazard_type',
        type=str,
        required=True,
        choices=['action_triggered', 'environmental'],
        help='Must be "action_triggered" or "environmental"'
    )
    parser.add_argument(
        '--detector_name',
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Thinking",
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=24,
    )
    args = parser.parse_args()
    meta_file_path = os.path.join(args.hazard_type, "annotation_info_pre.json")
    save_path = os.path.join(args.hazard_type, "annotation_info.json")
    save_folder = os.path.join(args.hazard_type, "annotate_image")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    verify_hazard(meta_file_path, save_path, args.detector_name, args.hazard_type, args.max_workers)