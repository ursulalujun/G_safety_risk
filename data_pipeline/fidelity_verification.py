import torch
from PIL import Image
import hpsv2
from transformers import BlipProcessor, BlipForQuestionAnswering

class ImageValidator:
    def __init__(self, device=None):
        """
        初始化验证器，加载 HPSv2 和 VQA 模型。
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在使用设备: {self.device} 加载模型...")

        # 1. 加载 HPS v2.1 (用于通用质量评分)
        # HPSv2 会自动下载模型 checkpoints
        print("正在加载 HPS v2.1...")
        self.hps_model_id = "hpsv2.1" 

        # 2. 加载 VQA 模型 (用于物理/逻辑检测 - VQAScore 核心)
        # 这里使用 BLIP-VQA，它轻量且在 VQA 任务上表现稳定
        # 如果你有 24G+ 显存，可以换成 LLaVA 或 Qwen-VL 以获得更强的推理能力
        print("正在加载 VQA 模型 (BLIP-VQA)...")
        self.vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(self.device)
        self.vqa_model.eval()
        
        print("所有模型加载完毕。")

    def score_hps(self, image_path, prompt="high quality image, realistic"):
        """
        第一层：使用 HPS v2.1 进行打分
        """
        # HPSv2 库通常接受图片路径列表
        try:
            # result 是一个包含分数的列表
            result = hpsv2.score([image_path], prompt, hps_version="v2.1")
            score = float(result[0])
            return score
        except Exception as e:
            print(f"HPS 打分出错: {e}")
            return -1.0

    def check_physics_vqa(self, image_path):
        """
        第二层：使用 VQA 进行物理和常识检测 (VQAScore 逻辑)
        定义一系列负面检测问题，如果回答是肯定的，说明有问题。
        """
        try:
            raw_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return {"passed": False, "reason": f"无法打开图片: {e}"}

        # 定义检查清单：(问题, 期望的回答)
        # 我们可以检查 "Yes" 的概率，这里简化为直接看模型生成的回答
        checklist = [
            {
                "question": "Is there any object floating in the air without support?",
                "bad_answer": "yes",
                "error_msg": "检测到物体悬浮 (Floating Object)"
            },
            {
                "question": "Is the image distorted or deformed?",
                "bad_answer": "yes",
                "error_msg": "检测到严重畸变 (Distortion)"
            },
            {
                "question": "Do the objects have realistic relative sizes?",
                "bad_answer": "no",
                "error_msg": "比例可能失调 (Unrealistic Scale)"
            },
            {
                "question": "Does the person typically have extra limbs or fingers?",
                "bad_answer": "yes",
                "error_msg": "肢体结构错误 (Bad Anatomy)"
            }
        ]

        issues = []
        
        for item in checklist:
            question = item["question"]
            inputs = self.vqa_processor(raw_image, question, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.vqa_model.generate(**inputs, max_new_tokens=20)
                answer = self.vqa_processor.decode(out[0], skip_special_tokens=True).lower().strip()
            
            # 简单的逻辑判定
            # 注意：BLIP 有时会回答 "yes it is" 等，所以用 in 判定
            if item["bad_answer"] in answer:
                issues.append(f"{item['error_msg']} (Q: {question}, A: {answer})")

        if len(issues) > 0:
            return {"passed": False, "reason": "; ".join(issues)}
        else:
            return {"passed": True, "reason": "符合物理常识"}

    def validate_image(self, image_path, hps_threshold=0.26):
        """
        主流程：结合 HPS 和 VQA
        hps_threshold: 经验值，HPSv2.1 > 0.26 通常质量尚可，要求高可设为 0.28+
        """
        print(f"\n>>> 正在检查: {image_path}")

        # --- 第一步：HPS 粗筛 ---
        hps_score = self.score_hps(image_path)
        print(f"HPS v2.1 分数: {hps_score:.4f}")
        
        if hps_score < hps_threshold:
            return {
                "status": "REJECTED",
                "stage": "HPS_Filter",
                "score": hps_score,
                "detail": "美学/通用质量分数过低"
            }

        # --- 第二步：VQA 物理精筛 ---
        vqa_result = self.check_physics_vqa(image_path)
        
        if not vqa_result["passed"]:
            return {
                "status": "REJECTED",
                "stage": "Physics_Filter",
                "score": hps_score,
                "detail": vqa_result["reason"]
            }

        # --- 通过 ---
        return {
            "status": "ACCEPTED",
            "score": hps_score,
            "detail": "图片质量合格且符合常识"
        }

# ================= 使用示例 =================
if __name__ == "__main__":
    # 实例化验证器 (加载一次模型，多次使用)
    validator = ImageValidator()

    # 假设你有一些图片路径
    # 请确保目录下有真实的图片文件
    test_images = ["test_floating.jpg", "test_good.jpg", "test_bad_hand.jpg"]

    # 模拟运行 (由于这里无法读取你本地文件，请替换为真实路径)
    # 这里仅做逻辑演示，你需要手动准备几张图来测
    import os
    
    # 创建一个伪造的测试环境，或者你可以直接修改上面的 list
    # 为了演示代码能跑，我先注释掉实际执行部分，写出调用逻辑
    
    """
    for img_path in test_images:
        if os.path.exists(img_path):
            result = validator.validate_image(img_path, hps_threshold=0.25)
            print(f"最终结果: {result['status']}")
            print(f"详情: {result['detail']}")
            print("-" * 30)
        else:
            print(f"找不到文件: {img_path}")
    """
    
    print("\n提示：请将 'test_images' 列表替换为你真实的图片路径即可运行检测。")
