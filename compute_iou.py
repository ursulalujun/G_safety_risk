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

from evaluation import SafetyEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file_path',
        type=str
    )
    args = parser.parse_args()

    evaluator = SafetyEvaluator(model_name="checkpoints/Qwen3-VL-32B-Instruct", img_save_folder="./")

    # 加载数据
    with open(args.file_path, 'r', encoding='utf-8') as f:
        gt_dataset = json.load(f)

    iou_list= []

    try:
        for i, gt_data in enumerate(gt_dataset["details"]):
            if gt_data["evaluation_metrics"]["safe_acc"] == 1 and gt_data["evaluation_metrics"]["risk_match"] == 1:
                pred_bbox_list = gt_data["evaluation_metrics"]["pred_bbox"]
                gt_bbox_list = gt_data["evaluation_metrics"]["gt_bbox"]
                iou = evaluator.compute_list_iou(gt_bbox_list, pred_bbox_list)
                gt_data["evaluation_metrics"]["iou_wo_label"]=iou
                iou_list.append(iou)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current results...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()
    finally:
        # final_metrics = evaluator.get_averages()      
        avg_iou = np.mean(iou_list)
        # gt_dataset["summary_metrics"]["avg_iou_wo_label"] = avg_iou

        save_path = args.file_path.replace("evaluation_results", "evaluation_results_iou")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(gt_dataset, f, indent=4, ensure_ascii=False)

        print(f"Avg IoU without label: {avg_iou:.4f} in {len(iou_list)} samples")