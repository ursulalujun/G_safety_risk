import argparse
import json
import os
import base64
from tqdm import tqdm
from openai import OpenAI  # Recommended SDK for calling Qwen-VL APIs
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import proxy_off

class FidelityVerifier:
    def __init__(self, model_name):
        """
        Initialize the verifier and configure the OpenAI-compatible API client.
        """
        api_key = os.environ['VERIFY_API_KEY']
        base_url = os.environ['VERIFY_API_URL']
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name
        print(f"Qwen-VL API Verifier initialized.")

    def check_physics_vqa(self, image_path):
        """
        Perform a comprehensive physical and common-sense inspection using Qwen3-VL's 
        reasoning capabilities in a single API call.
        """
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            return {"passed": False, "reason": f"Failed to read image: {e}"}

        # Constructing the system and user prompts for the Vision-Language Model
        system_prompt = (
            "You are a professional AI Image Quality Auditor. "
            "Analyze the image for physical logic, structural integrity, and biological accuracy."
        )
        
        user_content = [
            {
                "type": "text",
                "text": (
                    "Please inspect this image for the following quality issues:\n"
                    "1. Floating Objects: Are there objects hovering without support or logical contact?\n"
                    "2. Distortion: Is the image or its geometry distorted, deformed, or melted?\n"
                    "3. Unrealistic Scale: Are relative sizes of objects illogical (e.g., a giant cat)?\n"
                    "4. Bad Anatomy: Does the person have extra limbs, fused fingers, or broken joints?\n\n"
                    "5. Residual Bounding Boxes: Are there any unremoved red bounding boxes visible in the image?\n\n"
                    "Output Format: For each issue found, provide: [Error Message] - [Suggestion (Point error categoty and give refinement suggestion)]. "
                    "If the image is physically consistent and has no issues, output only 'PASSED'."
                )
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
            )
            
            analysis_result = response.choices[0].message.content
            
            # Logic: If 'PASSED' is in the response and the response is short, consider it valid.
            if "thinking" in self.model_name.lower():
                analysis_result = analysis_result.split("</think>")[-1]
            if "PASSED" in analysis_result.upper():
                return {"passed": True, "reason": "Consistent with common sense"}
            else:
                return {"passed": False, "reason": analysis_result.strip()}

        except Exception as e:
            print(f"API Call Error: {e}")
            return {"passed": False, "reason": f"API Error: {str(e)}"}

    def validate_image(self, image_path):
        """
        Main workflow: Use Qwen-VL to screen the image for fidelity issues.
        """
        vqa_result = self.check_physics_vqa(image_path)
        
        if not vqa_result["passed"]:
            return f"REJECTED: {vqa_result['reason']}"

        return "ACCEPTED"

def process_single_item(validator, item):
    """
    Helper function to process one JSON item. 
    This is the target function for the ThreadPoolExecutor.
    """
    risk = item.get("safety_risk")
    if risk is None:
        return item
        
    img_path = risk.get("edit_image_path")

    if img_path and os.path.exists(img_path):
        result = validator.validate_image(img_path)
        if result['status'] == 'REJECTED':
            risk['fidelity_check'] = f"REJECTED: {result['detail']}"
        else:
            risk['fidelity_check'] = "ACCEPTED"
    else:
        risk['fidelity_check'] = "ERROR: Image not found"
    
    return item

def verify_fidelity(meta_file_path, save_path, hazard_type, max_workers):
    """
    Load JSON data, process images through the verifier, and save results.
    """
    proxy_off()
    validator = FidelityVerifier()
    
    with open(meta_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Starting verification for {hazard_type}...")

    updated_data = []

    import ipdb; ipdb.set_trace()
    process_single_item(validator, data[0])

    # Use ThreadPoolExecutor for concurrent API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map the function to the data
        future_to_item = {executor.submit(process_single_item, validator, item): item for item in data}
        
        with tqdm(total=len(data), desc="🖼️ Processing Images") as pbar:
            for future in as_completed(future_to_item):
                try:
                    # Collect the result as they complete
                    result_item = future.result()
                    updated_data.append(result_item)
                except Exception as e:
                    print(f"\nWorker error: {e}")
                finally:
                    pbar.update(1)

    # Save the updated annotations back to a JSON file
    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Verification complete. Results saved to {save_path}")

# ================= Execution Entry Point =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Fidelity Verification using Qwen-VL API")
    parser.add_argument(
        '--hazard_type', 
        type=str, 
        required=True, 
        help='The category of hazard to process'
    )
    parser.add_argument(
        '--max_workers', 
        type=int, 
        default=24,
    )

    args = parser.parse_args()
    
    # Define file paths based on the hazard type provided
    meta_file_path = os.path.join(args.hazard_type, "edition_info.json")
    save_path = os.path.join(args.hazard_type, "annotation_info.json")
    verify_fidelity(meta_file_path, save_path, args.hazard_type, args.max_workers)