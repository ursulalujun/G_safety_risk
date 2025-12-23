import argparse
import json
import torch
import os
import openai
from PIL import Image, ImageDraw
from utils import visualize_bbox, image_to_base64

# 假设你使用的是 HuggingFace transformers 库中的实现
# 如果是官方 repo，请根据官方 API 调整 import
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import AutoModelForCausalLM, AutoTokenizer

OBJECT_SELECT_TEMPLATE="""
I have edited an image to synthesize an {hazard_type} safety hazard based on the following context:

- Safety Principle (to be violated): {safety_principle}{instruction}
- Edit Description: '{edit_desc}'

Task: GroundingDINO has detected multiple bounding boxes for the label '{label}' (marked with indices 0, 1, etc.). You need to identify which specific object corresponds to the newly edited content and acts as the primary trigger of the safety hazard.

Reasoning criteria: Analyze the object state and spatial relationships. The correct object should be the one creating the risk described in the safety principle (e.g., placed too close to water/fire, or blocking a path), distinct from other safe background objects.

Output Requirement: Return ONLY the integer index number.
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
  "final_verdict": "PASS" | "FAIL", // PASS if all scores > 7 and risk is clear
  "refinement_suggestion": "If FAIL, provide a specific instruction to fix it (e.g., 'Move the cup closer to the edge' or 'Add steam to the water')."
}}
```
"""

class HazardVerifier:
    def __init__(self, verifier_model, selector_model):
        print("Loading GroundingDINO...")
        self.processor = AutoProcessor.from_pretrained(verifier_model)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(verifier_model).to("cuda")
        key = os.getenv("ANNOTATION_API_KEY")
        url = os.getenv("ANNOTATION_API_URL")
        self.client = openai.OpenAI(api_key=key, base_url=url)
        self.selector = selector_model

    def detect(self, image, text_prompts, box_threshold=0.35, text_threshold=0.25):
        """
        返回格式: List of (box, score, label)
        """
        inputs = self.processor(images=image, text=text_prompts, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 后处理结果
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )[0]
        
        return results
    
    def select_best_box(self, image, boxes, label, risk_info, hazard_type):
        """
        当存在多个 bbox 时，调用 Qwen 进行选择。
        思路：将带有 bbox 标注的图片传给 Qwen，让它结合 context 判断。
        """

        bbox_list = []
        for bbox in boxes:
            bbox_list.append({
                "label": label,
                "bounding_box": bbox
            })
        image_with_box = visualize_bbox(image, bbox_list)
        base64_image = image_to_base64(image_with_box)
        
        instruction = risk_info.get("instruction", "")
        edit_desc = risk_info.get("edit_description", "")
        safety_principle = risk_info.get("safety_principle", {})
        if instruction != "":
            ins_context = f"\n- Action Instruction: {instruction}\n"
        else:
            ins_context = ""
        
        prompt = OBJECT_SELECT_TEMPLATE.format(hazard_type=hazard_type, safety_principle=safety_principle, edit_desc=edit_desc, instruction=ins_context, label=label)

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
            model=self.selector,
            messages=messages,
        ).choices[0].message.content
        import ipdb; ipdb.set_trace()
        
        try:
            selected_index = int(''.join(filter(str.isdigit, response)))
            # 加上边界检查
            if 0 <= selected_index < len(boxes):
                return selected_index
        except:
            print(f"Warning: Qwen returned unclear answer: {response}. Defaulting to 0.")
        
        return 0
    
    def verify_state(self, image, risk_info, hazard_type):
        base64_image = image_to_base64(image)
        
        instruction = risk_info.get("instruction", "")
        edit_desc = risk_info.get("edit_description", "")
        safety_principle = risk_info.get("safety_principle", {})
        hazard_objects = risk_info.get("Hazard_related_area", {})
        if instruction != "":
            ins_context = f"\n- Action Instruction: {instruction}\n"
        else:
            ins_context = ""
        
        prompt = STATE_CHECK_TEMPLATE.format(hazard_type=hazard_type, hazard_objects=hazard_objects, safety_principle=safety_principle, edit_desc=edit_desc, instruction=ins_context)
        import ipdb; ipdb.set_trace()

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
            model=self.selector,
            messages=messages,
        ).choices[0].message.content
        import ipdb; ipdb.set_trace()

        if "PASS" in response: 
            return True
        else:
            return False

def verify_hazard(meta_file_path, save_path, hazard_type, max_workers):
    verifier = HazardVerifier("IDEA-Research/grounding-dino-base", "Qwen/Qwen3-VL-235B-A22B-Thinking")

    with open(meta_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    error_list = []
    for item in data:
        scene_type = item["scene_type"]
        
        print(f"\nProcessing scene: {scene_type}")

        risk = item["safety_risk"]
        
        if risk is None:
            continue

        image_path = risk["edit_image_path"]
    
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}, skipping...")
            continue
            
        pil_image = Image.open(image_path).convert("RGB")

        objects_to_detect = []
        
        hazard_objs = risk.get("Hazard_related_area", {})
        if hazard_type.lower() == "environmental":
            risk["bbox_annotation"] = {}
            for obj_name in hazard_objs:
                objects_to_detect.append(("target_object", obj_name))
        else:
            target_objs = hazard_objs.get("target_object")
            constraint_objs = hazard_objs.get("constraint_object")
            
            risk["bbox_annotation"] = {
                "target_object": {},
                "constraint_object": {}
            }

            if target_objs is not None:
                for target_obj_name in target_objs:
                    objects_to_detect.append(("target_object", target_obj_name))
            if constraint_objs is not None:
                for constraint_obj_name in constraint_objs:
                    objects_to_detect.append(("constraint_object", constraint_obj_name))

        # --- first step: annotate bbox ---
        for role, obj_label in objects_to_detect:
            # GroundingDINO 需要英文 prompt，最好加上 '.' 
            text_prompt = obj_label + "." 
            results = verifier.detect(pil_image, text_prompts=[text_prompt])
            
            boxes = results["boxes"]
            scores = results["scores"]
            
            if len(boxes) == 0:
                print(f"  [Not Found] {role}: {obj_label}")
                continue
            
            final_box = None
            
            if boxes is None or len(boxes)==0:
                error_risk = item.copy()
                error_risk["error_info"]=f"[Missing Hazard-Related Object] {role}: {obj_label}"
                error_list.append(error_risk)
            elif len(boxes) == 1:
                final_box = boxes[0].tolist()
                print(f"  [Single Hit] {role}: {obj_label} detected at {final_box}")
                risk["bbox_annotation"][role][obj_label]=final_box
            else:
                print(f"  [Ambiguity] {role}: {obj_label} has {len(boxes)} candidates. Asking Qwen...")
                best_idx = verifier.select_best_box(pil_image, boxes, obj_label, risk, hazard_type)
                final_box = boxes[best_idx]
                print(f"  [Qwen Selected] Index {best_idx} for {obj_label}")
                risk["bbox_annotation"][role][obj_label]=final_box

        # --- second step: check spatial relationship ---
        verifier.verify_state(pil_image, risk, hazard_type)
    
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    return error_list

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
        '--max_workers', 
        type=int,
        default=24,
    )
    args = parser.parse_args()
    meta_file_path = os.path.join(args.hazard_type, "edition_info.json")
    save_path = os.path.join(args.hazard_type, "annotation_info.json")
    verify_hazard(meta_file_path, save_path, args.hazard_type, args.max_workers)
