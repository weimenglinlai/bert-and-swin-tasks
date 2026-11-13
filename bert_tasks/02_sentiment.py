# -*- coding: utf-8 -*-
"""
任务二 · 任务一（2）：中文情感分析（sentiment）

对应参考文档：
  五、实践步骤及要求 (一) 4.实践任务 (1) 中文情感分析（sentiment）

要点：
  在 Hugging Face 国内镜像站（https://hf-mirror.com）搜索并下载**其他已微调**的
  中文情感分析 BERT 模型，本地加载后对给定中文语句判断“正面 / 中性 / 负面”。

选用模型：clapAI/modernBERT-base-multilingual-sentiment
  选用理由：该模型是 BERT 系（ModernBERT）的情感**三分类**微调模型，原生标签就是
            negative / neutral / positive，与题目要求的“正面 / 中性 / 负面”一一对应，
            不需要额外做标签映射；本地实测在中文语句上三类的判定均正确。
            （选型时另外实测了 lxyuan/distilbert-…、jackietung/bert-base-chinese-…、
              ccutwuxiao/bert-chinese-sentiment-c3、tabularisai/… 等多个模型，
              它们的实测表现记录在 models/_candidates/ 备选目录中。）
  本地目录：models/sentiment-3class
"""
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging as hf_logging

# 关闭 transformers 的加载进度条与冗余提示，让控制台输出干净、便于结果输出
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 中文标签映射（Hugging Face 标准情感三分类标签）
LABEL_ZH = {
    "positive": "正面",
    "neutral": "中性",
    "negative": "负面",
}

# 待测试的中文语句（覆盖正面 / 负面 / 中性三类）
SENTENCES = [
    "这家餐厅的菜太好吃了，服务也特别周到，下次一定还来！",
    "今天心情特别好，事情都很顺利。",
    "今天下了一整天的雨，出门还摔了一跤，真是糟糕透顶。",
    "这台电脑用了不到一周就经常死机，客服还一直推脱，很失望。",
    "会议定于下午三点在行政楼二楼会议室召开。",
    "这是一篇关于机器学习模型压缩方法的学术论文。",
]


def print_banner(subtask):
    """输出个人信息（说明截图要求截图中有个人信息）"""
    print("=" * 70)
    print("：        ：")
    print("：20                 说明：危自福")
    print("：AI")
    print("任务二：大语言模型和视觉大模型")
    print("子任务：" + subtask)
    print("=" * 70)


def main():
    print_banner("基于 Bert 的文本处理应用 —— 中文情感分析（sentiment）")

    model_path = os.path.join(BASE_DIR, "models", "sentiment-3class")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n本地模型目录：", model_path)
    print("推理设备：", device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    print("\n模型配置中的标签：" + str(model.config.id2label))
    print("\n" + "-" * 70)
    print("情感分析结果")
    print("-" * 70)

    for i, text in enumerate(SENTENCES, 1):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

        pred_id = int(torch.argmax(probs).item())
        pred_en = model.config.id2label[pred_id]
        pred_zh = LABEL_ZH.get(pred_en, pred_en)

        print(f"\n[{i}] 语句：{text}")
        print(f"    判定结果：{pred_zh}（{pred_en}）    置信度：{probs[pred_id].item():.2%}")
        print("    各类别概率：" + "  ".join(
            f"{LABEL_ZH.get(model.config.id2label[j], model.config.id2label[j])}={probs[j].item():.2%}"
            for j in range(len(probs))
        ))

    print("\n" + "-" * 70)
    print("情感分析完成。")


if __name__ == "__main__":
    main()
