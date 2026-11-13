# -*- coding: utf-8 -*-
"""
任务二 · 任务一（3）：中文命名实体识别（NER）

对应参考文档：
  五、实践步骤及要求 (一) 4.实践任务 (2) 中文命名实体识别（NER）

要点：
  在 Hugging Face 国内镜像站（https://hf-mirror.com）搜索并下载**其他已微调**的
  中文命名实体识别 BERT 模型，本地加载后识别给定中文语句中的实体。

选用模型：shibing624/bert4ner-base-chinese
  该模型在中文 NER 语料上对 bert-base-chinese 做了 TokenClassification 微调，
  标签体系为 PER（人名）/ LOC（地名）/ ORG（机构名），
  本地目录：models/ner

【BIOES 标注体系说明】（参考文档要求）
  B（Begin）  实体开头        I（Inside） 实体中间部分
  O（Outside）非实体          E（End）    实体结尾
  S（Single） 单字符构成的实体
  本脚本使用 pipeline 的 aggregation_strategy="simple"，
  引擎会自动把 B-/I-/E-/S- 等子标签合并为完整实体并给出实体类型。
"""
import os

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import logging as hf_logging

# 关闭 transformers 的加载进度条与冗余提示，让控制台输出干净、便于结果输出
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 实体类型中文对照
TYPE_ZH = {
    "PER": "人名",
    "PERSON": "人名",
    "LOC": "地名",
    "GPE": "地名",
    "ORG": "机构名",
    "TIME": "时间",
}

# 待识别的中文语句
SENTENCES = [
    "2024年5月，在参加了由腾讯公司举办的AI研讨会。",
    "李彦宏在深圳市的百度大厦发布了新一代大模型，随后又赶赴北京参加世界互联网大会。",
    "鲁迅原名周树人，是浙江绍兴人，代表作有《呐喊》和《彷徨》。",
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
    print_banner("基于 Bert 的文本处理应用 —— 中文命名实体识别（NER）")

    model_path = os.path.join(BASE_DIR, "models", "ner")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n本地模型目录：", model_path)
    print("推理设备：", device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    print("\n模型标签集合：" + str(list(model.config.id2label.values())))
    print("\n" + "-" * 70)
    print("命名实体识别结果")
    print("-" * 70)

    for i, text in enumerate(SENTENCES, 1):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()

        # 用 word_ids 把 subword 预测结果还原成整词：
        # 同一个 word 可能被切成多个 subword（如 2024 → 202 + ##4），
        # 这些子词的标签以第一个子词为准，文字需要完整拼接，不能丢弃。
        word_ids = inputs.word_ids()
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        entities = []
        cur = None
        prev_wid = None
        for idx, wid in enumerate(word_ids):
            if wid is None:                     # [CLS]/[SEP] 等特殊符号
                if cur:
                    entities.append(cur)
                    cur = None
                prev_wid = None
                continue

            label = model.config.id2label[pred_ids[idx]]
            token = tokens[idx]
            txt = token[2:] if token.startswith("##") else token

            if wid == prev_wid:                 # 同一个词的后继子词
                if cur is not None:
                    cur["text"] += txt
                continue
            prev_wid = wid

            if label == "O" or label.startswith("O-"):
                if cur:
                    entities.append(cur)
                    cur = None
                continue

            prefix, _, etype = label.partition("-")
            if prefix in ("B", "S") or cur is None or cur["type"] != etype:
                if cur:
                    entities.append(cur)
                cur = {"text": txt, "type": etype}
            else:                               # I / E 续接
                cur["text"] += txt
            if prefix in ("E", "S"):            # 实体到此结束
                entities.append(cur)
                cur = None
        if cur:
            entities.append(cur)

        print(f"\n[{i}] 语句：{text}")
        if entities:
            for ent in entities:
                zh = TYPE_ZH.get(ent["type"], ent["type"])
                print(f"    {ent['text']:<12s} → {ent['type']}（{zh}）")
        else:
            print("    未识别到实体")

    print("\n" + "-" * 70)
    print("命名实体识别完成。")


if __name__ == "__main__":
    main()
