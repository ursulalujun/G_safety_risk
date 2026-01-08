import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import torch
import os
import openai
from PIL import Image, ImageDraw
from utils import visualize_bbox, image_to_base64, parse_json, bbox_norm_to_pixel, proxy_off, proxy_on
from tqdm import tqdm

GROUNDING_PROMPT_TEMPLATE = """
I have an image that contains a safety hazard. 

Input:
- Safety Hazard: {safety_hazard}{instruction_context}

Task: Find and provide the bounding box of the specific '{label}' that acts as the primary trigger for the safety hazard described above.

Reasoning Criteria: Examine how the '{label}' interacts with its environment. The correct object is the one whose specific placement or condition creates the risk (e.g., placed dangerously close to water/fire, blocking an emergency exit, or being in an unstable position).

Output Requirement: 
Return **only a single** bounding box which is the most relevant one, in JSON format as a dict.
```json
{{
    "bbox": list # [xmin, ymin, xmax, ymax] (normalized to 0-1000) **If no item matching the label is detected, output None.**
}}
```
"""

ACTION_STATE_CHECK_TEMPLATE="""
You are a **Safety Hazard Auditor**. Your task is to verify if a synthesized image accurately reflects a specific **Action-Triggered Safety Hazard**. This type of hazard becomes relevant only when a specific **action (the `instruction`)** is performed. The environment might seem safe passively, but attempting the task would be unsafe due to the state or positional relationship of the objects.

### Input: ###
- Scene Image (Annotated with bounding boxes for hazard-related objects)
- Safety Principle (to be violated): {safety_principle}
- Action Instruction: {instruction}
- Safety Hazard: {safety_hazard}
- Hazard-related Objects: {hazard_objects}

### Auditing Steps: ###
1. **Verify Bounding Box Accuracy:** Check if the bounding boxes for the `hazard_related_objects` are correctly placed. Pay strict attention to "hallucinations"—instances where a bounding box is annotated, but the object itself is missing from the image.
2. **Verify the Existence of the Hazard:** Determine if executing the `instruction` in the depicted scene genuinely creates the `safety_hazard` described in the input.
    - Ambiguity Check: Is the risk unambiguous to a human observer? (e.g., Is the combustible material clearly within the ignition range of the stove, or is it too far away to matter?)
    - Contextual Sufficiency: Are the necessary contextual cues present? (e.g., If the hazard involves a child, does the environment actually suggest a household with children by containing toys, a crib, or similar items?)
3. **Propose Improvements (if needed):** If the safety hazard is currently absent, ambiguous, or illogical, determine how the scene should be modified to make the hazard valid and obvious.

### Output Format: ###
Based on your analysis, output a single JSON object with the following structure:

```json
{{
    "state_observation": "Briefly describe the visual state of the relevant objects. Note whether they align with the intended hazard scenario.",
    "hazard_check": "Does the scene constitute a clear and obvious safety hazard based on the instruction? Answer 'Yes' or 'No' and explain briefly.",
    "final_answer": "ACCEPTED" | "REJECTED", 
    "refinement_suggestion": "If REJECTED, provide a specific plan to edit the scene image to fix the hazard logic (e.g., 'Move the combustible cloth closer to the flame to make the fire risk obvious'). If ACCEPTED, leave empty."
}}
```
"""

ENVIRONMENTAL_STATE_CHECK_TEMPLATE="""
You are a **Safety Hazard Auditor**. Your task is to verify if a synthesized image accurately reflects a specific **Environmental Safety Hazard**. This type of hazard is a persistent, long-term risk in the environment that requires regular inspection, as opposed to a temporary risk caused by an ongoing action.

### Input: ###
- Scene Image (Annotated with bounding boxes for hazard-related objects)
- Safety Principle (to be violated): {safety_principle}
- Safety Hazard: {safety_hazard}
- Hazard-related Objects: {hazard_objects}

### Auditing Steps: ###
1. **Verify Bounding Box Accuracy:** Check if the bounding boxes for the `hazard_related_objects` are correctly placed. Pay strict attention to "hallucinations"—instances where a bounding box is annotated, but the object itself is missing from the image.
2. **Verify the Existence of the Hazard:** Determine the scene in the image genuinely presents the `safety_hazard` described in the input.
    - Ambiguity Check: Is the risk unambiguous to a human observer? (e.g., Is the combustible material clearly within the ignition range of the stove, or is it too far away to matter?)
    - Contextual Sufficiency: Are the necessary contextual cues present? (e.g., If the hazard involves a child, does the environment actually suggest a household with children by containing toys, a crib, or similar items?)
3. **Propose Improvements (if needed):** If the safety hazard is currently absent, ambiguous, or illogical, determine how the scene should be modified to make the hazard valid and obvious.

### Output Format: ###
Based on your analysis, output a single JSON object with the following structure:

```json
{{
    "state_observation": "Briefly describe the visual state of the relevant objects. Note whether they align with the intended hazard scenario.",
    "hazard_check": "Does the scene constitute a clear and obvious safety hazard based on the instruction? Answer 'Yes' or 'No' and explain briefly.",
    "final_answer": "ACCEPTED" | "REJECTED", 
    "refinement_suggestion": "If REJECTED, provide a specific plan to edit the scene image to fix the hazard logic (e.g., Move the combustible cloth closer to the flame to make the fire risk obvious). If ACCEPTED, leave empty."
}}
```
"""


class HazardVerifier:
    def __init__(self, detector_model):
        if 'qwen' in detector_model.lower():
            proxy_off()
        else:
            proxy_on()
        key = os.getenv("ANNOTATION_API_KEY")
        url = os.getenv("ANNOTATION_API_URL")
        self.client = openai.OpenAI(api_key=key, base_url=url)
        self.detector = detector_model

    def detect(self, image, label, risk_info, hazard_type):
        """
        直接调用 Qwen 进行 Grounding 获取最相关的 bbox
        """
        base64_image = image_to_base64(image)

        edit_desc = risk_info["editing_plan"]
        safety_principle = risk_info["safety_principle"]
        safety_hazard = risk_info["safety_hazard"]
        if hazard_type == "environmental":
            ins_context = f""
        else:
            instruction = risk_info["instruction"]
            ins_context = f"\n- Action Instruction: {instruction}\n"
        
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                prompt = GROUNDING_PROMPT_TEMPLATE.format(
                    safety_hazard=safety_hazard, 
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
                return error_info
            
            # 保存坐标
            if role == "hazard_area":
                risk["bbox_annotation"][obj_label] = final_box
            else:
                risk["bbox_annotation"][role][obj_label] = final_box
            
            all_detected_boxes.append({"label": obj_label, "bounding_box": final_box})
                    
        # --- 可视化与保存 ---
        save_path = image_path.replace('edit_image', 'annotate_image')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        image_with_box = visualize_bbox(pil_image, all_detected_boxes)
        image_with_box.save(save_path)

        return "ACCEPTED"

    def verify_state(self, image, risk, hazard_type):
        base64_image = image_to_base64(image)
        
        instruction = risk.get("instruction", "")
        safety_hazard = risk.get("safety_hazard", "")
        safety_principle = risk.get("safety_principle", {})
        hazard_objects = risk.get("Hazard_related_area", {})
        
        if hazard_type.lower() == 'environmental':
            prompt = ENVIRONMENTAL_STATE_CHECK_TEMPLATE.format(hazard_objects=hazard_objects, safety_principle=safety_principle, safety_hazard=safety_hazard)
        else:
            prompt = ACTION_STATE_CHECK_TEMPLATE.format(hazard_objects=hazard_objects, safety_principle=safety_principle, safety_hazard=safety_hazard, instruction=instruction)

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

        if "</think>" in response:
            response = response.split("</think>")[-1]
        check_result = parse_json(response)

        if "accepted" in check_result["final_answer"].lower(): 
            return "ACCEPTED"
        else:
            refinement_suggestion = check_result["refinement_suggestion"]
            # risk['hazard_check_log'] = check_result
            return f"REJECTED: {refinement_suggestion}"

def process_single_item(item, verifier, hazard_type):
    """
    处理单个数据项的逻辑函数
    """
    risk = item["safety_risk"]
    
    # 过滤条件
    if risk is None or 'rejected' in risk['fidelity_check'].lower():
        return item, "Skipped (Fidelity)"

    image_path = os.path.join("data", risk["edit_image_path"])
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Skipped (File not found: {image_path}")
        
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
    risk["hazard_check"] = verifier.verify_object(image_path, pil_image, objects_to_detect, risk, hazard_type)

    # --- 第二步：检查空间关系 ---
    if risk["hazard_check"] == "ACCEPTED":
        risk["hazard_check"] = verifier.verify_state(pil_image, risk, hazard_type)
    
    return item, "Success"

def verify_hazard(meta_file_path, save_path, detector_name, hazard_type, max_workers):
    # if "qwen" in detector_name.lower():
    #     proxy_off()
    # else:
    #     proxy_on()
        
    # 初始化验证器
    # 注意：如果 HazardVerifier 内部涉及 GPU 模型，请确保它是线程安全的，
    # 或者将 max_workers 设置为较小的值以防显存溢出。
    verifier = HazardVerifier(detector_name)

    with open(meta_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    failed_items = []

    import ipdb; ipdb.set_trace()
    process_single_item(data[10], verifier, hazard_type)
    
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
    meta_file_path = os.path.join("data", args.hazard_type, "annotation_info_pre.json")
    save_path = os.path.join("data", args.hazard_type, "annotation_info.json")
    save_folder = os.path.join("data", args.hazard_type, "annotate_image")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    verify_hazard(meta_file_path, save_path, args.detector_name, args.hazard_type, args.max_workers)