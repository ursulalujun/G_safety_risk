import os
import json
import argparse
import traceback
import threading
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from nodes.editing_planner import EditingPlanner
from nodes.fidelity_verifier import FidelityVerifier
from nodes.hazard_verifier import HazardVerifier
from nodes.scene_editor import SceneEditor

from utils import extract_and_plot_principles, proxy_off, proxy_on, extract_principle_id
from principle_tracker import PrincipleTracker

class RiskWeaverPipeline:
    def __init__(self, args):
        self.args = args
        self.max_retries = args.max_retries

        # Initialize PrincipleTracker for balanced generation
        checkpoint_path = os.path.join("data", args.hazard_type, "principle_checkpoint.json")
        self.principle_tracker = PrincipleTracker(
            max_per_principle=args.max_per_principle,
            checkpoint_path=checkpoint_path
        )

        # Initialize Agents with principle tracker
        self.planner = EditingPlanner(args.planner_model, principle_tracker=self.principle_tracker)

        # Determine if the editor is local
        is_local_editor = os.path.exists(args.editor_model)
        self.editor = SceneEditor(args.editor_model, local_model=is_local_editor)

        self.fidelity_critic = FidelityVerifier(args.fidelity_model)
        self.hazard_detective = HazardVerifier(args.detector_model)

        # Thread-safe lock for principle counting
        self._stop_flag = False

    def process_image(self, image_path, scene_type):
        """
        Implements the Verify-Refine Loop for a single image.
        """
        # Check if all principles have reached quota
        if not self.principle_tracker.is_principle_available(self.args.hazard_type):
            print(f"✅ All safety principles have reached the maximum quota ({self.args.max_per_principle})")
            print("Stopping pipeline...")
            with threading.Lock():
                self._stop_flag = True
            return None

        meta_info = {image_path: scene_type}

        # 1. Risk Architect (EditingPlanner) - Planning
        try:
            editing_item = self.planner.generate_edit_plan(image_path, self.args.hazard_type, meta_info)
            # Check if planning failed due to no available principles
            if editing_item is None or editing_item.get('safety_risk') is None:
                return None
        except Exception as e:
            print(f"[-] Planning failed for {image_path}: {e}")
            return None

        current_feedback = None
        for attempt in range(0, self.max_retries):
            # 2. Scene Editor Agent - Execution
            try:
                edited_item = self.editor.edit_scene(editing_item, self.args.hazard_type,
                                                     current_feedback, attempt)
            except Exception as e:
                print(f"[-] Scene Editing failed for {image_path}: {e}")
                return None

            # 3. Dual Verification Mechanism
            # 3a. Physics Critic (FidelityVerifier)
            edit_img_path = edited_item['safety_risk'].get('edit_image_path')
            fidelity_res = self.fidelity_critic.validate_image(edit_img_path)
            if "REJECTED" in fidelity_res:
                current_feedback = f"Image Fidelity Error, Original Editing Plan: {risk_data['editing_plan']} Refinement Suggestion{fidelity_res.split('REJECTED')[-1]}"
                print(f"[!] Attempt {attempt} Rejected for {os.path.basename(image_path)}: {current_feedback}")
                risk_data[f"feedback_{attempt}"] = current_feedback
                continue

            # 3b. Risk Detective (HazardVerifier)
            from PIL import Image
            risk_data = edited_item["safety_risk"]
            pil_img = Image.open(edit_img_path).convert("RGB")

            # Prepare objects to detect based on hazard type
            objects_to_detect = self._get_objects_to_detect(risk_data)
            detective_res = self.hazard_detective.verify_object(edit_img_path, pil_img, objects_to_detect, risk_data, self.args.hazard_type)

            if "REJECTED" in detective_res:
                current_feedback = f"{detective_res.split('REJECTED:')[-1]}, Original Editing Plan: {risk_data['editing_plan']} Refinement Suggestion: add missing objects."
                risk_data[f"feedback_{attempt}"] = current_feedback
                continue

            annotate_img_path = edit_img_path.replace('edit_image', 'annotate_image')
            anno_pil_img = Image.open(annotate_img_path).convert("RGB")
            state_res = self.hazard_detective.verify_state(anno_pil_img, risk_data, self.args.hazard_type)

            # 4. Decision: Pass Both Checks?
            if "REJECTED" in state_res:
                current_feedback = f"Hazard Simulation Error, Refinement Suggestion{state_res.split('REJECTED')[-1]}"
                risk_data[f"feedback_{attempt}"] = current_feedback
            else:
                edited_item["state"] = "successful"
                # Increment principle counter after successful generation
                risk_data = editing_item["safety_risk"]
                safety_principle_text = risk_data.get("safety_principle", "")
                principle_id = extract_principle_id(safety_principle_text)
                if principle_id is not None:
                    self.principle_tracker.increment(self.args.hazard_type, principle_id)
                return edited_item

        print(f"[Failed!] {os.path.basename(image_path)}: {current_feedback}")
        edited_item["state"] = "failed"
        return edited_item

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
    parser.add_argument('--planner_model', type=str, default='gemini-2.5-pro')
    parser.add_argument('--editor_model', type=str, default="gpt-image-1-mini") # "gemini-2.5-flash-image"
    parser.add_argument('--detector_model', type=str, default="Qwen/Qwen3-VL-235B-A22B-Thinking")
    parser.add_argument('--fidelity_model', type=str, default="Qwen/Qwen3-VL-235B-A22B-Thinking")
    parser.add_argument('--max_workers', type=int, default=16)
    parser.add_argument('--max_retries', type=int, default=3)
    parser.add_argument('--max_per_principle', type=int, default=50,
                        help='Maximum samples per safety principle (default: 50)')
    args = parser.parse_args()

    if os.path.exists(args.editor_model):
        args.max_workers = 1

    # Setup directories
    root_folder = os.path.join("data", "base_image")
    meta_path = os.path.join("data", "meta_info.json")
    output_path = os.path.join("data", args.hazard_type, "annotated_data.json")

    for folder_name in ["check_image", "edit_image", "annotate_image"]:
        save_folder = os.path.join("data", args.hazard_type, folder_name)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

    with open(meta_path, 'r') as f:
        meta_dict = json.load(f)

    # Gather image paths
    image_tasks = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.png')):
                full_path = os.path.join(dirpath, filename)
                if full_path in meta_dict:
                    image_tasks.append((full_path, meta_dict[full_path]))

    pipeline = RiskWeaverPipeline(args)
    final_results = []

    print(f"🚀 Starting Risk-Weaver Pipeline with {args.max_workers} workers. Processing {len(image_tasks)} images in total...")
    print(f"📊 Maximum {args.max_per_principle} samples per safety principle")
    proxy_on()

    image_tasks = image_tasks[:200]
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_path = {
                executor.submit(pipeline.process_image, path, stype): path
                for path, stype in image_tasks
            }

            with tqdm(total=len(image_tasks), desc="Processing Pipeline") as pbar:
                for future in as_completed(future_to_path):
                    # Check if we should stop
                    if pipeline._stop_flag:
                        print("🛑 Stop flag detected, cancelling remaining tasks...")
                        # Cancel remaining futures
                        for f in future_to_path:
                            if not f.done():
                                f.cancel()
                        break

                    result = future.result()
                    if result is not None:
                        final_results.append(result)
                    pbar.update(1)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current results...")
    except Exception as e:
        print(f"Fatal Error: {e}")
        traceback.print_exc()

    # Save final JSON output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    extract_and_plot_principles(os.path.join("data", args.hazard_type), final_results)
    print(f"✅ Pipeline complete. Generated {len(final_results)} samples. Saved to {output_path}")

if __name__ == "__main__":
    main()