# python image_edition.py --hazard_type action_triggered --max_workers 1
import argparse
import base64
import cv2
from diffusers import QwenImageEditPipeline
from io import BytesIO
import json
import openai
import os
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import random
import re
import requests
import time
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import calculate_diff_bbox, visualize_bbox, parse_base64_image

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

key = os.getenv("EDIT_API_KEY")
url = os.getenv("EDIT_API_URL")

client = openai.OpenAI(api_key=key, base_url=url)

crucial_rules = """### Crucial Rules: ###
1.  **Edit Within Bounding Box:** The red bounding box in the input image defines the inpainting mask. Perform edits within this area.
2.  **Follow Edition Plan Exactly:** You must **strictly adhere** to every detail provided in the **Edition PLAN** (textual, colors, size, materials, spatial relationship, etc.).
3.  **Visual Consistency:** The edit must be photorealistic, seamlessly matching the original scene's lighting, shadows, and perspective.
4.  **Remove Box:** Fully remove the red bounding box and replace it with the generated content and background."""

# - **Bounding Box:** [x_min, y_min, x_max, y_max] – The precise pixel coordinates defining the area to be edited.
ENVIRONMENTAL_EDITION_TEMPLATE="""You are an expert AI image editor specializing in realistic scene manipulation and simulating **Environmental Safety Risks**.

You are provided with an input image containing a **red bounding box**, along with a specific safety principle, an edition plan, and an risk-related area. Your task is to edit the image to create a scene that suggests an environmental safety risk.

{crucial_rules}

### Output: ###
Return only the final edited image.

Your input:
- Safety Principle: {safety_principle}
- Edition Plan: {edition_plan}
- Risk-related Area: {hazard_related_area}{feedback}"""

# ### Inputs: ###
# - **Image:** The source image containing the red bounding box annotation.
# - **Safety Principle:** The specific safety standard to be violated, provided as "[ID]. [Description]".
# - **Edition Plan:** A precise instruction detailing the visual modification required.
# - **Risk-related Area:** The specific area where a safety risk exists, or the visual cue identifying a risk in the environment.{feedback_param_info}

ACTION_TRIGGERED_EDITION_TEMPLATE="""You are an expert AI image editor specializing in realistic scene manipulation and simulating **Action-Triggered Safety Risks**.

You are provided with an input image containing a **red bounding box**, along with a specific safety principle, an edition plan, and an risk-related area. Your task is to edit the image to create a scene that suggests an action-triggered safety risk. This means the generated scene might appear benign passively, but becomes actively dangerous when a human attempts to perform the specific action defined in the **Instruction** input.

{crucial_rules}

### Output: ###
Return only the final edited image.

Your input:
- Safety Principle: {safety_principle}
- Instruction: {instruction}
- Edition Plan: {edition_plan}
- Risk-related Area {hazard_related_area}{feedback}"""

# ### Inputs: ###
# - **Image:** The source image containing the red bounding box annotation.
# - **Safety Principle:** The specific safety standard to be violated, provided as "[ID]. [Description]".
# - **Instruction:** The specific human task or action that makes the generated situation dangerous.
# - **Edition Plan:** A precise instruction detailing the visual modification required.
# - **Risk-related Area:** The specific area where a safety risk exists, or the visual cue identifying a risk in the environment.{feedback_param_info}

# simple template for open-source edition model
SIMPLE_TEMPLATE="""{edition_plan}
**Notice:** The red bounding box in the input image is your "inpainting mask" or area for edition. Please completely remove the red bounding box in your edited image."""

class SceneEditor:
    def __init__(self, editor_model, local_model=False):
        self.local_model = local_model
        if local_model:
            self.pipeline = QwenImageEditPipeline.from_pretrained(editor_model)
            print("pipeline loaded")
            self.pipeline.to(torch.bfloat16)
            self.pipeline.to("cuda")
            self.pipeline.set_progress_bar_config(disable=None)
        else:
            key = os.getenv("PLAN_API_KEY")
            url = os.getenv("PLAN_API_URL")
            self.client = openai.OpenAI(api_key=key, base_url=url)
            self.editor = editor_model

    def edit_scene(self, edited_item, hazard_type, feedback, iter_num=0, max_retries=3):
        risk = edited_item['safety_risk']
        safety_principle = risk['safety_principle']
        edition_plan = risk['edition_plan']
        hazard_related_area = risk['hazard_related_area']
        if iter_num > 0 and feedback is not None:
            image_path = risk['edit_image_path']
            save_path = image_path.replace(f'{iter_num-1}.png', f'{iter_num}.png')
            feedback_param_info = "\n- **Feedback:** Critique from the previous iteration, strictly formatted as **`[Error type], [Refinement Suggestion]`**. **This is the highest authority.** If the `[Refinement Suggestion]` conflicts with the **Edition Plan**, you must override the Plan and strictly follow the Feedback.\n"
            feedback = f"\n- Feedback: {feedback}"
        else:
            image_path = risk['pre_image_path']
            save_path = image_path.replace('check_image', 'edit_image')[:-4]+'__'+f'{iter_num}.png'
            feedback_param_info=""
            feedback=""
        if not os.path.exists(image_path):
            print(f"[ERROR]: {image_path} not find image!")
            return edited_item
        
        # skip the existing image
        # if os.path.exists(save_path):
        #     risk['edit_image_path']=save_path
            # _, risk['edition_bbox']=calculate_diff_bbox(image_path, save_path, 80)
            # return edited_item
        
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        
        if hazard_type.lower()=="action_triggered":
            instruction = risk['instruction']
            prompt = ACTION_TRIGGERED_EDITION_TEMPLATE.format(safety_principle=safety_principle, 
                                                        edition_plan=edition_plan,
                                                        instruction=instruction, 
                                                        hazard_related_area=hazard_related_area,
                                                        crucial_rules=crucial_rules,
                                                        # feedback_param_info=feedback_param_info,
                                                        feedback=feedback) 
        else:
            prompt = ENVIRONMENTAL_EDITION_TEMPLATE.format(safety_principle=safety_principle, 
                                                        edition_plan=edition_plan, 
                                                        hazard_related_area=hazard_related_area,
                                                        crucial_rules=crucial_rules,
                                                        # feedback_param_info=feedback_param_info,
                                                        feedback=feedback)
        
        if not self.local_model:
            messages=[
                {
                    "role": "user",
                    "content":[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": { "url": f"data:image/png;base64,{image_base64}" }
                        }
                    ],
                }
            ]
            
            for attempt in range(1, max_retries + 1):
                try:
                    if 'gpt' in self.editor.lower():
                        result = self.client.images.edit(
                            # "gpt-image-1" 0.16
                            model=self.editor,
                            image=[
                                open(image_path, "rb"),
                            ],
                            prompt=prompt
                        )
                        image_base64 = result.data[0].b64_json
                        image_bytes = base64.b64decode(image_base64)
                    else:
                        response = self.client.chat.completions.create(
                            model=self.editor,
                            messages=messages
                        )
                        image_base64 = response.choices[0].message.content # '![image](data:image/png;base64,iVBORw0K...)'
                        # img_data = image_base64.split("base64,")[1]
                        # img_data = img_data[:-1].strip()
                        img_data = parse_base64_image(image_base64)
                        image_bytes = base64.b64decode(img_data)
                except Exception as e:
                    print(f"⚠️ [Attempt {attempt}/{max_retries}] 处理失败: {image_path} | Error: {e}")
                    if attempt < max_retries: # response.choices[0].finish_reason != "content_filter" and 
                        time.sleep(1)  
                    else:
                        raise e 
        else:
            prompt = SIMPLE_TEMPLATE.format(edition_plan=edition_plan)
            image = Image.open(image_path).convert("RGB")
            inputs = {
                "image": image,
                "prompt": prompt,
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 50,
            }

            with torch.inference_mode():
                output = self.pipeline(**inputs)
                output_image = output.images[0]
                # output_image.save("hazard_output.png")
                buffered = BytesIO()
                output_image.save(buffered, format="PNG") 
                img_binary_data = buffered.getvalue() 
                base64_str = base64.b64encode(img_binary_data).decode('utf-8')
                image_bytes = base64.b64decode(base64_str)

        risk['edit_image_path']=save_path
        
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))

        with open(save_path, "wb") as f:
            f.write(image_bytes)
        
        # try:
        #     image_gen_resized, risk['edition_bbox']=calculate_diff_bbox(image_path, save_path, 80)
        #     cv2.imwrite(save_path, image_gen_resized)
        # except Exception as e:
        #     print(f"Image error: {save_path}")
            
        return edited_item


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
        '--model_name', 
        type=str, 
        default='../checkpoints/Qwen-Image-Edit' # 'gemini-2.5-flash-image',
    )
    parser.add_argument(
        '--editor_model', 
        type=int, 
        default=1,
    )
    args = parser.parse_args()

    with open(f'{args.hazard_type}/edition_plan.json', 'r') as f:
        edition_plan = json.load(f)
    
    edit_folder = os.path.join(args.hazard_type, "edit_image")
    if not os.path.exists(edit_folder):
        os.mkdir(edit_folder)

    if os.path.exists(args.editor_model):
        local_flag = True
    else:
        local_flag = False
    editor = SceneEditor(args.editor_model, local_flag)
    
    import ipdb; ipdb.set_trace()
    editor.edit_scene(edition_plan[0], args.hazard_type)

    results = [None] * len(edition_plan)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_index = {
            executor.submit(editor.edit_scene, plan, args.hazard_type): i 
            for i, plan in enumerate(edition_plan)
        }

        with tqdm(total=len(edition_plan), desc="🖼️ 处理图像") as pbar:
            for future in as_completed(future_to_index):
                idx = future_to_index[future] 
                try:
                    res = future.result()
                    results[idx] = res 
                except Exception as e:
                    print(f"❌ 索引 {idx} 处理出错: {e}")
                    # results[idx] = {"error": str(e), "status": "failed"} 
                finally:
                    pbar.update(1)

    print("✅ 所有图片编辑完成！")

    results = [r for r in results if r is not None] 
    with open(f'{args.hazard_type}/edition_info.json', 'w') as f:
        json.dump(results, f, indent=2)
    