# proxy管理
# 加feedback，改prompt
import os
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from nodes.edition_planner import EditionPlanner
from nodes.fidelity_verifier import FidelityVerifier
from nodes.hazard_verifier import HazardVerifier
from nodes.scene_editor import SceneEditor

class RiskWeaverPipeline:
    def __init__(self, args):
        self.args = args
        self.max_retries = args.max_retries
        
        # Initialize Agents
        self.planner = EditionPlanner(args.planner_model)
        
        # Determine if the editor is local
        is_local_editor = os.path.exists(args.editor_model)
        self.editor = SceneEditor(args.editor_model, local_model=is_local_editor)
        
        self.fidelity_critic = FidelityVerifier(args.fidelity_model)
        self.hazard_detective = HazardVerifier(args.detector_model)

    def process_image(self, image_path, scene_type):
        """
        Implements the Verify-Refine Loop for a single image.
        """
        meta_info = {image_path: scene_type}
        current_feedback = None
        
        for attempt in range(0, self.max_retries):
            # 1. Risk Architect (EditionPlanner) - Planning
            try:
                # We pass current_feedback to the planner if it's a retry (Reflector logic)
                edition_item = self.planner.generate_edit_plan(image_path, self.args.hazard_type, meta_info)
                if edition_item['safety_risk'] is None:
                    return edition_item
            except Exception as e:
                print(f"[-] Planning failed for {image_path} at attempt {attempt}: {e}")
                continue

            # 2. Scene Editor Agent - Execution
            try:
                edited_item = self.editor.edit_scene(edition_item, self.args.hazard_type, 
                                                     current_feedback, attempt)
            except Exception as e:
                print(f"[-] Scene Edition failed for {image_path}: {e}")
                continue

            # 3. Dual Verification Mechanism
            # 3a. Physics Critic (FidelityVerifier)
            edit_img_path = edited_item['safety_risk'].get('edit_image_path')
            fidelity_res = self.fidelity_critic.validate_image(edit_img_path)
            if "REJECTED" in fidelity_res:
                current_feedback = f"Image Fidelity Error, Refinement Suggestion: {fidelity_res.split("REJECTED")[-1]}"
                print(f"[!] Attempt {attempt} Rejected for {os.path.basename(image_path)}: {current_feedback}")
                continue

            # 3b. Risk Detective (HazardVerifier)
            # This updates edited_item['safety_risk']['hazard_check'] internally
            # We simulate the process_single_item logic here
            from PIL import Image
            risk_data = edited_item["safety_risk"]
            pil_img = Image.open(edit_img_path).convert("RGB")
            
            # Prepare objects to detect based on hazard type
            objects_to_detect = self._get_objects_to_detect(risk_data)
            detective_res = self.hazard_detective.verify_object(edit_img_path, pil_img, objects_to_detect, risk_data, self.args.hazard_type)
            
            if "REJECTED" in detective_res:
                current_feedback = f"Hazard Area Detection Error, Refinement Suggestion: {detective_res.split("REJECTED")[-1]}"
                risk_data[f"feedback_{attempt}"] = current_feedback
                # print(f"[!] Attempt {attempt} Rejected for {os.path.basename(image_path)}: {current_feedback}")
                continue
            
            state_res = self.hazard_detective.verify_state(pil_img, risk_data, self.args.hazard_type)

            # 4. Decision: Pass Both Checks?
            if "REJECTED" in state_res:
                # Reflector Agent / Feedback Mechanism
                # Collect reasons for failure to pass back to the Planner in next loop
                current_feedback = f"Hazard Simulation Error: {state_res.split("REJECTED")[-1]}"
                risk_data[f"feedback_{attempt}"] = current_feedback
                # print(f"[!] Attempt {attempt} Rejected for {os.path.basename(image_path)}: {current_feedback}")
            else:
                return edited_item
        
        print(f"[Failed!] {os.path.basename(image_path)}: {current_feedback}")
        return None # Failed after max retries

    def _get_objects_to_detect(self, risk_data):
        objects = []
        hazard_objs = risk_data["hazard_related_area"]
        if self.args.hazard_type == "environmental":
            risk_data["bbox_annotation"] = {}
            for obj_name in hazard_objs:
                objects.append(("hazard_area", obj_name))
        else:
            risk_data["bbox_annotation"] = {"target_object": {}, "constraint_object": {}}
            for name in hazard_objs.get("target_object", []):
                objects.append(("target_object", name))
            for name in hazard_objs.get("constraint_object", []):
                objects.append(("constraint_object", name))
        return objects

def main():
    parser = argparse.ArgumentParser(description="Risk-Weaver Multi-agent Pipeline")
    parser.add_argument('--hazard_type', type=str, required=True, choices=['action_triggered', 'environmental'])
    parser.add_argument('--planner_name', type=str, default='Qwen/Qwen3-VL-235B-A22B-Thinking')
    parser.add_argument('--editor_model_path', type=str, required=True, help="Path to local model or API name")
    parser.add_argument('--detector_name', type=str, default="Qwen/Qwen3-VL-235B-A22B-Thinking")
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--max_retries', type=int, default=3)
    args = parser.parse_args()

    # Setup directories
    root_folder = os.path.join("data", args.hazard_type, "base_image")
    meta_path = os.path.join("data", args.hazard_type, "meta_info.json")
    output_path = os.path.join("data", args.hazard_type, "annotated_data.json")

    for folder_name in ["check_image", "edit_image", "annotate_image"]:
        save_folder = os.path.join("data", args.hazard_type, folder_name)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

    with open(meta_path, 'r') as f:
        meta_dict = json.load(f)

    # Gather image paths
    image_tasks = []
    for filename in os.listdir(root_folder):
        if filename.lower().endswith(('.jpg', '.png')):
            full_path = os.path.join(root_folder, filename)
            if full_path in meta_dict:
                image_tasks.append((full_path, meta_dict[full_path]))

    pipeline = RiskWeaverPipeline(args)
    final_results = []

    print(f"🚀 Starting Risk-Weaver Pipeline with {args.max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_path = {
            executor.submit(pipeline.process_image, path, stype): path 
            for path, stype in image_tasks
        }
        
        with tqdm(total=len(image_tasks), desc="Processing Pipeline") as pbar:
            for future in as_completed(future_to_path):
                result = future.result()
                if result:
                    final_results.append(result)
                pbar.update(1)

    # Save final JSON output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Pipeline complete. Generated {len(final_results)} samples. Saved to {output_path}")

if __name__ == "__main__":
    main()