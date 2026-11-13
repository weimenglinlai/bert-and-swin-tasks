# -*- coding: utf-8 -*-
"""
任务二 · 任务一（4）：中文句子相似度计算（Sentence Similarity）

对应参考文档：
  五、实践步骤及要求 (一) 4.实践任务 (3) 中文句子相似度计算（Sentence Similarity）

要点：
  在 Hugging Face 国内镜像站（https://hf-mirror.com）搜索并下载预训练 BERT 模型，
  将输入语句转换为句向量后，用**余弦相似度**计算两条语句的语义相似程度。
  参考文档推荐的模型：shibing624/text2vec-base-chinese
  本地目录：models/text2vec-base-chinese

实现说明：
  优先用 sentence-transformers 按模型自带池化配置取句向量；
  若该模型缺少 sentence-transformers 配置文件，则回退为
  「BERT 编码 + Mean Pooling」手工取句向量，同样满足“语句转向量 + 余弦相似度”的要求。
"""
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers import logging as hf_logging

# 关闭 transformers 的加载进度条与冗余提示，让控制台输出干净、便于结果输出
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 待比较的语句对
SENTENCE_PAIRS = [
    ("我喜欢吃苹果", "我爱吃苹果"),
    ("今天天气很好", "今天的天气真不错"),
    ("怎么申请学校的奖学金", "奖学金申请流程是什么"),
    ("共享单车的需求预测与调度", "共享单车潮汐现象与车辆调度优化"),
    ("今天的天气真不错", "这个苹果吃起来又甜又脆"),
    ("怎么申请学校的奖学金", "图书馆几点关门"),
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


def mean_pooling(last_hidden_state, attention_mask):
    """Mean Pooling：对所有 token 的隐藏状态按真实长度取平均"""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class Text2Vec:
    """句向量编码器（优先 sentence-transformers，缺失配置时回退 Mean Pooling）"""

    def __init__(self, model_path, device):
        self.device = device
        self.backend = None

        try:
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer(model_path, device=device)
            self.backend = "sentence-transformers（按模型自带池化配置）"
            return
        except Exception as e:
            print("sentence-transformers 加载失败，回退到 Mean Pooling：", type(e).__name__)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        self.backend = "transformers + Mean Pooling"

    def encode(self, sentences):
        if self.backend.startswith("sentence-transformers"):
            vecs = self.st.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
            return vecs

        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        vecs = mean_pooling(out.last_hidden_state, inputs["attention_mask"])
        return F.normalize(vecs, p=2, dim=1)


def cosine_sim(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main():
    print_banner("基于 Bert 的文本处理应用 —— 中文句子相似度计算（Sentence Similarity）")

    model_path = os.path.join(BASE_DIR, "models", "text2vec-base-chinese")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n本地模型目录：", model_path)
    print("推理设备：", device)

    encoder = Text2Vec(model_path, device)
    print("句向量后端：", encoder.backend)

    print("\n" + "-" * 70)
    print("句子相似度计算结果（余弦相似度，取值 0~1，越接近 1 越相似）")
    print("说明：text2vec-base-chinese 的余弦值整体偏紧凑——完全相同为 1.0，")
    print("      同义改写约 0.85~0.98，主题相关约 0.6~0.85，无关语句约 0.3，")
    print("      因此分档阈值按该模型的实际分布设定为 0.85 / 0.6 / 0.4。")
    print("-" * 70)

    for i, (s1, s2) in enumerate(SENTENCE_PAIRS, 1):
        v1, v2 = encoder.encode([s1]), encoder.encode([s2])
        sim = cosine_sim(v1[0], v2[0])

        if sim >= 0.85:
            level = "高度相似"
        elif sim >= 0.6:
            level = "较为相关"
        elif sim >= 0.4:
            level = "弱相关"
        else:
            level = "不相似"

        print(f"\n[{i}] 语句A：{s1}")
        print(f"    语句B：{s2}")
        print(f"    余弦相似度：{sim:.4f}    →  {level}")

    print("\n" + "-" * 70)
    print("句子相似度计算完成。")


if __name__ == "__main__":
    main()
