# -*- coding: utf-8 -*-
"""
任务二 · 任务一（1）：BERT 本地模型加载基础测试

对应参考文档：
  四、实践内容 (一) 基于 Bert 的文本处理应用
  五、实践步骤及要求 (一) 1.模型下载 / 2.环境配置 / 3.代码编辑和运行

要点：
  先按参考文档图 2.3~2.6 在 Hugging Face 国内镜像站（https://hf-mirror.com）
  搜索并下载 bert-base-chinese 的全部文件，放入 models/bert-base-chinese/；
  再按参考文档图 2.7 把 from_pretrained() 的参数换成该本地文件夹路径，
  实现完全离线的本地加载。
"""
import os

from transformers import BertTokenizer, BertModel
from transformers import logging as hf_logging
from torch import no_grad

# 关闭 transformers 的加载进度条与冗余提示，让控制台输出干净、便于结果输出
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

# 本脚本所在目录（保证从任意工作目录运行都能找到本地模型）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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
    print_banner("基于 Bert 的文本处理应用 —— BERT 本地模型加载基础测试")

    # 指向本地模型文件夹路径（参考文档图 2.7：把模型名称替换为本地文件夹路径）
    model_path = os.path.join(BASE_DIR, "models", "bert-base-chinese")
    print("\n本地模型目录：", model_path)

    # 从本地加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)

    # 使用示例
    text = "你好，世界"
    inputs = tokenizer(text, return_tensors="pt")  # 输出为 PyTorch 张量
    print("输入文本:", text)
    print("Input IDs:", inputs["input_ids"])
    print("Attention Mask:", inputs["attention_mask"])
    print()

    # 推理（关闭梯度以提高性能和减少内存）
    with no_grad():
        outputs = model(**inputs)

    # 提取输出
    last_hidden_state = outputs.last_hidden_state    # [batch, seq_len, hidden_size]
    pooled_output = outputs.pooler_output            # [batch, hidden_size]
    cls_embedding = last_hidden_state[:, 0, :]       # [batch, hidden_size], 即 [CLS] 向量

    # 输出形状信息
    print("Last hidden state shape:", last_hidden_state.shape)        # [1, sequence_length, 768]
    print("Pooled output shape (sentence vector):", pooled_output.shape)  # [1, 768]
    print("CLS embedding [last_hidden_state[:, 0, :]] shape:", cls_embedding.shape)  # [1, 768]
    print()

    # 输出前几个数值（便于对比）
    print("Pooled output (first 10 elements):", pooled_output[0, :10].numpy())
    print("CLS embedding (first 10 elements):", cls_embedding[0, :10].numpy())

    # 小技巧：查看每个 token 的位置
    print("\n【小技巧】分词后每个 token 的位置：")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    for i, token in enumerate(tokens):
        print(f"Position {i}: {token}")


if __name__ == "__main__":
    main()
