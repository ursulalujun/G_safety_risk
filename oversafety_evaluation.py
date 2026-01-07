import argparse
import base64
import torch
import time
import json
import numpy as np
import re
import os
import traceback
from PIL import Image

from evaluation import SafetyAgent, SafetyEvaluator

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
        '--target_model', 
        type=str # "/mnt/shared-storage-user/ai4good1-share/models/Qwen3-VL-32B-Instruct"
    )
    parser.add_argument(
        '--evaluation_model', 
        type=str, 
        default='Qwen/Qwen3-VL-235B-A22B-Thinking',
    )
    args = parser.parse_args()

    DATASET_PATH = os.path.join("data_pipeline", "data", args.hazard_type, "annotated_data.json")
    save_folder = os.path.join("results", args.hazard_type, os.path.basename(args.target_model))
    OUTPUT_FILE = os.path.join(save_folder, f'oversafety_evaluation_results.json')
    os.makedirs(save_folder, exist_ok=True)

    # 初始化
    agent = SafetyAgent(model_name=args.target_model) 
    # evaluator = SafetyEvaluator(model_name=args.evaluation_model, img_save_folder=save_folder)

    # 加载数据
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        gt_dataset = json.load(f)# [:100]
    
    print(f"Start evaluating {len(gt_dataset)} samples...")

    detailed_logs = []

    try:
        for i, gt_data in enumerate(gt_dataset):
            if gt_data['safety_risk'] is None:
                continue
            dr = gt_data['safety_risk']
            image_path = os.path.join("data_pipeline", gt_data['image_path'])
            instruction = dr.get("instruction", "") 

            if not os.path.exists(image_path):
                print(f"skipped_image_not_found: {image_path}")
                # detailed_logs.append({
                #     "id": i,
                #     "image": image_path,
                #     "status": "skipped_image_not_found"
                # })
                continue

            prediction, raw_text = agent.infer(image_path, instruction, args.hazard_type)
            print(f"Prediction: {prediction}")
            oversafety = not prediction['safe']
            # res = evaluator.evaluate(prediction, gt_data, image_path)
            # print(f"  Metrics -> Acc: {res['safe_acc']})

            log_entry = {
                "id": i,
                "image_path": image_path,
                "model_output_raw": raw_text,       # 模型的原始文本输出 (可能含Thinking Process)
                "model_output_json": prediction,    # 解析后的JSON
                "oversafety": oversafety           # 评测结果 (whether oversafety)
            }
            detailed_logs.append(log_entry)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current results...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()
    finally:
        # final_metrics = evaluator.get_averages()
        
        oversafety_num = 0
        for log in detailed_logs:
            if log['oversafety']:
                oversafety_num += 1
        oversafety_rate = oversafety_num/len(detailed_logs)

        final_output_data = {
            "summary_metrics": oversafety_rate,
            "details": detailed_logs
        }

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_output_data, f, indent=4, ensure_ascii=False)

        print("\n=== Final Aggregated Metrics ===")
        print(f"Avg Oversafety Rate: {oversafety_rate:.4f}")
            
        print(f"Saved summary and detailed logs to {OUTPUT_FILE}")