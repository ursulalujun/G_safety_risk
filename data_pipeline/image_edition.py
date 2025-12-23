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
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import calculate_diff_bbox, visualize_bbox

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

key = os.getenv("OPENAI_API_KEY")
url = os.getenv("OPENAI_API_URL")

client = openai.OpenAI(api_key=key, base_url=url)

crucial_rules = """
### Crucial Rules: ###
1.  **Target Identification:** Locate the red bounding box in the input image. This is your "inpainting mask" or area of focus.
2.  **Strict Content Generation:** Inside the target area, generate the object described by the **Object Label**. You must **strictly adhere** to every detail provided in the **Edit Description**. Precisely match the specified size, shape, materials, textures, colors, physical state (e.g., wet, fragile, unstable), and exact spatial relationships with surrounding environmental elements.
3.  **Visual Consistency:**
    - **Photorealism:** The generated object must look indistinguishable from a real photo.
    - **Lighting & Shadows:** Match the lighting direction, color temperature, and intensity of the original room. Cast realistic shadows onto the surrounding surfaces.
    - **Perspective:** Ensure the object's size and angle are consistent with the camera's perspective.
4. Enhanced Visual Salience: Make appropriate visual modifications to ensure the safety risk is prominent and clearly conveyed. For example, 
    - Falling Object Hazard: Items must be placed on surfaces that appear high, unstable, or precariously close to the edge.
    - Improper Chemical/Medicine Storage: The outer packaging of the chemical or medicine must have clear, legible text or icons explicitly identifying it as a chemical, cleaning agent, or medication.
    - Power Off Before Cleaning/Moving: The device must have explicit visual cues indicating it is currently on or active, such as an illuminated indicator light, a visible display screen showing content, or being clearly plugged into an outlet with the cable taut.
    - Flammable Items Near Heat: the added paper towel must look like it is dangerously close to the burner
5.  **Cleanup:** The final output must **NOT** show the red bounding box. The red lines must be completely removed and replaced by the generated object and the background behind it.
"""

ENVIRONMENTAL_HAZARD_TEMPLATE="""
You are an expert AI image editor specializing in realistic scene manipulation and safety simulation.

You are provided with an input image containing a **red bounding box**, along with a specific safety principle, an edit description, and an object label. Your task is to edit the image **strictly within the area defined by the red bounding box**. You must generate realistic visual content that replaces the red box and the area inside it, seamlessly blending with the surrounding environment.

### Inputs: ###
- **Image:** The source image containing the red bounding box annotation.
- **Safety Principle:** The specific safety standard to be violated, provided as "[ID]. [Description]".
- **Edit Description:** A precise instruction detailing the visual modification required.
- **Object Label:** The category or name of the object to be added or manipulated.

{crucial_rules}

### Output: ###
Return only the final edited image.

### Example Usage (for your reference) ###

    **Input Parameters provided to the model:**

    - **Safety Principle:** "3. Flammable Items Near Heat: Keep all flammable items at least three feet away from heat sources."
    - **Edit Description:** "Add a free-standing paper towel roll on the granite countertop, placing it dangerously close to the rear-left stove burner."
    - **Object Label:** "Paper Towel Roll"
    - **Image:** [Image containing a red box drawn on the counter next to the stove]

    **Expected Model Behavior:**
    The model reads the image, identifies the red box next to the stove, generates a realistic roll of paper towels in that exact spot (matching the lighting of the kitchen), ensures it looks "too close" to the stove, removes the red outline, and outputs the clean, edited photo.

Your input:
- Safety Principle: {safety_principle}
- Edit Description: {edit_description}
- Object Label: {label}
"""

ACTION_TRIGGERED_HZARD_TEMPLATE="""
You are an expert AI image editor specializing in realistic scene manipulation and simulating **Action-Triggered Safety Hazards**.

You are provided with an input image containing a **red bounding box**, along with a specific safety principle, a human instruction (task), an edit description, and an object label. Your task is to edit the image **strictly within the area defined by the red bounding box**. You must generate a "Action-Triggered Hazard." This means the generated situation might appear benign passively, but becomes actively hazardous *only* when a human attempts to perform the specific action defined in the **Instruction** input. The visual cues you generate must imply why performing that instruction is now unsafe.

### Inputs: ###
- **Image:** The source image containing the red bounding box annotation.
- **Safety Principle:** The specific safety standard to be violated, provided as "[ID]. [Description]".
- **Instruction:** The specific human task or action that makes the generated situation hazardous (e.g., "Wipe the table," "Open the cabinet").
- **Edit Description:** A precise instruction detailing the visual modification required.
- **Object Label:** The category or name of the object to be added or manipulated.

{crucial_rules}

### Output: ###
Return only the final edited image.

### Example Usage (for your reference) ###

    **Input Parameters provided to the model:**
    - **Safety Principle:** "3. Flammable Items Near Heat: Keep all flammable items at least three feet away from heat sources."
    - **Instruction:** "Cook the ramen noodles."
    - **Edit Description:** "Add a free-standing paper towel roll on the granite countertop, placing it dangerously close to the rear-left stove burner."
    - **Object Label:** "Paper Towel Roll"
    - **Image:** [Image containing a red box drawn on the counter next to the stove]

    **Expected Model Behavior:**
    The model reads the image, identifies the red box next to the stove, generates a realistic roll of paper towels in that exact spot (matching the lighting of the kitchen), ensures it looks "too close" to the stove, removes the red outline, and outputs the clean, edited photo.

Your input:
- Safety Principle: {safety_principle}
- Instruction: {instruction}
- Edit Description: {edit_description}
- Object Label: {label}
"""

SIMPLE_TEMPLATE="""
{edit_description}
**Notice:** The red bounding box in the input image is your "inpainting mask" or area for edition. Please completely remove the red bounding box in your edited image.
"""

def edit_image(edit_plan, model, hazard_type):
    
    # - **Bounding Box:** [x_min, y_min, x_max, y_max] – The precise pixel coordinates defining the area to be edited.
    
    image_idx = 0

    if edit_plan['safety_risk'] is None:
        return 
    
    risk = edit_plan['safety_risk']
    safety_principle = risk['safety_principle']
    edit_description = risk['edit_description']
    label = risk['label']
    image_path = risk['pre_image_path']
    if not os.path.exists(image_path):
        print(f"[ERROR]: {image_path} not find image!")
        return

    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    # bbox_2d = risk['pre_bbox_2d']

    # img = Image.open(image_path)
    # width, height = img.size

    # abs_y1 = int(bbox_2d[1] / 1000 * height)
    # abs_x1 = int(bbox_2d[0] / 1000 * width)
    # abs_y2 = int(bbox_2d[3] / 1000 * height)
    # abs_x2 = int(bbox_2d[2] / 1000 * width)

    # if abs_x1 > abs_x2:
    #     abs_x1, abs_x2 = abs_x2, abs_x1

    # if abs_y1 > abs_y2:
    #     abs_y1, abs_y2 = abs_y2, abs_y1

    # bbox_2d=[abs_x1, abs_y1, abs_x2, abs_y2]

    if hazard_type.lower()=="action_triggered":
        instruction = risk['instruction']
        prompt = ACTION_TRIGGERED_HZARD_TEMPLATE.format(safety_principle=safety_principle, 
                                                    edit_description=edit_description,
                                                    instruction=instruction, 
                                                    label=label,
                                                    crucial_rules=crucial_rules) 
    else:
        prompt = ENVIRONMENTAL_HAZARD_TEMPLATE.format(safety_principle=safety_principle, 
                                                    edit_description=edit_description, 
                                                    label=label,
                                                    crucial_rules=crucial_rules)
    
    if type(model) is str:
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
        
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        image_base64 = response.choices[0].message.content # '![image](data:image/png;base64,iVBORw0K...)'
        img_data = image_base64.split("base64,")[1]
        img_data = img_data[:-1].strip()
        image_bytes = base64.b64decode(img_data)
    else:
        prompt = SIMPLE_TEMPLATE.format(edit_description=edit_description)
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
            output = model(**inputs)
            output_image = output.images[0]
            # output_image.save("hazard_output.png")
            buffered = BytesIO()
            output_image.save(buffered, format="PNG") 
            img_binary_data = buffered.getvalue() 
            base64_str = base64.b64encode(img_binary_data).decode('utf-8')
            image_bytes = base64.b64decode(base64_str)

    save_path = image_path.replace('check_image', 'edit_image')[:-4]+'__'+str(image_idx)+'.png'
    risk['edit_image_path']=save_path
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))

    with open(save_path, "wb") as f:
        f.write(image_bytes)
    
    image_gen_resized, risk['edition_bbox']=calculate_diff_bbox(image_path, save_path, 80)
    cv2.imwrite(save_path, image_gen_resized)

    image_idx += 1
        
    return edit_plan


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
        '--max_workers', 
        type=int, 
        default=24,
    )
    args = parser.parse_args()

    with open(f'{args.hazard_type}/edition_plan.json', 'r') as f:
        edit_plan = json.load(f)

    if 'Qwen' in args.model_name:
        pipeline = QwenImageEditPipeline.from_pretrained(args.model_name)
        print("pipeline loaded")
        pipeline.to(torch.bfloat16)
        pipeline.to("cuda")
        pipeline.set_progress_bar_config(disable=None)
        model = pipeline
    else:
        model = args.model_name

    # edit_image(edit_plan[0], model, args.hazard_type)

    results = [None] * len(edit_plan)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_index = {
            executor.submit(edit_image, plan, model, args.hazard_type): i 
            for i, plan in enumerate(edit_plan)
        }

        with tqdm(total=len(edit_plan), desc="🖼️ 处理图像") as pbar:
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
    