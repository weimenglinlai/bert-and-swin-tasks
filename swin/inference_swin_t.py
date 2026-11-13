# -*- coding: utf-8 -*-
"""
任务二 · 任务二（1）：基于 Swin-Transformer 的图像分类（Swin-T 模型）

对应参考文档：
  五、实践步骤及要求 (二) 基于Swin-Transformer的图像分类 → 3.代码编写和运行

说明：
  本脚本代码与参考文档给的 inference.py 一致，仅做了三处必要调整：
    (1) 路径改为相对本文件定位，避免换工作目录后找不到文件；
    (2) 开头输出个人信息（说明截图要求）；
    (3) 一次可预测多张测试图片。
"""
import os
import sys
import json
import time
import warnings

import torch
import yaml
from PIL import Image
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, "Swin-Transformer")
# 保证可以像参考文档那样直接导入官方模型代码
sys.path.insert(0, REPO_DIR)

# 屏蔽与推理无关的告警（如 torch.meshgrid 的弃用提示），让控制台输出更干净
warnings.filterwarnings("ignore")

# 从官方代码导入Swin模型
from models.swin_transformer import SwinTransformer

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 测试图片（ImageNet-1K 中同时包含拉布拉多犬与藏獒两类）
TEST_IMAGES = [
    os.path.join(BASE_DIR, "images", "labrador.jpg"),
    os.path.join(BASE_DIR, "images", "tibetan_mastiff.jpg"),
]


def print_banner():
    """输出个人信息（说明截图要求截图中有个人信息）"""
    print("=" * 70)
    print("：        ：")
    print("：20                 说明：危自福")
    print("：AI")
    print("任务二：大语言模型和视觉大模型")
    print("子任务：基于 Swin-Transformer 的图像分类（Swin-T 模型）")
    print("=" * 70)


def load_model(weights_path, config_path):
    """
    从配置文件加载模型参数，并加载预训练权重
    参数:
        weights_path (str): 预训练权重文件路径
        config_path (str): 模型配置文件路径
    返回:
        torch.nn.Module: 加载权重后的模型
        dict: 模型配置信息
    """
    # 加载配置文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 提取配置参数
    model_cfg = config['MODEL']
    swin_cfg = model_cfg['SWIN']

    # 从文件名提取图像尺寸（配置文件未明确指定时）
    img_size = int(model_cfg['NAME'].split('_')[-1])

    # 根据SwinTransformer类定义初始化模型
    model = SwinTransformer(
        img_size=img_size,
        patch_size=4,  # 从文件名可知patch_size=4
        in_chans=3,  # 默认RGB输入
        num_classes=1000,  # ImageNet类别数
        embed_dim=swin_cfg['EMBED_DIM'],
        depths=swin_cfg['DEPTHS'],
        num_heads=swin_cfg['NUM_HEADS'],
        window_size=swin_cfg['WINDOW_SIZE'],
        mlp_ratio=4.,  # 类定义默认值
        qkv_bias=True,  # 类定义默认值
        qk_scale=None,  # 类定义默认值
        drop_rate=0.,  # 类定义默认值
        attn_drop_rate=0.,  # 类定义默认值
        drop_path_rate=model_cfg['DROP_PATH_RATE'],  # 从配置文件加载
        norm_layer=torch.nn.LayerNorm,  # 类定义默认值
        ape=False,  # 类定义默认值
        patch_norm=True,  # 类定义默认值
        use_checkpoint=False,  # 类定义默认值
        fused_window_process=False  # 类定义新增参数，使用默认值
    )

    # 加载预训练权重
    checkpoint = torch.load(weights_path, map_location=device)

    # 兼容不同格式的权重文件
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()  # 设置为评估模式

    # 计算模型参数量
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 返回模型和配置信息
    return model, {
        'img_size': img_size,
        'model_name': model_cfg['NAME'],
        'window_size': swin_cfg['WINDOW_SIZE'],
        'param_count': param_count
    }


def load_class_names(json_path):
    """加载ImageNet类别名称映射"""
    with open(json_path, 'r') as f:
        class_idx = json.load(f)
    return {int(k): v[1] for k, v in class_idx.items()}


def preprocess_image(image_path, img_size):
    """根据模型输入尺寸预处理图像"""
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)


def predict(image_path, model, config, class_names, top_k=5):
    """预测图片类别并返回结果"""
    # 预处理图片
    image = preprocess_image(image_path, config['img_size'])

    # 推理并计时
    start_time = time.time()
    with torch.no_grad():  # 关闭梯度计算
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    inference_time = time.time() - start_time

    # 获取前k个预测结果
    top_prob, top_idx = torch.topk(probabilities, k=top_k)
    top_prob = top_prob.squeeze().cpu().numpy()
    top_idx = top_idx.squeeze().cpu().numpy()

    # 整理结果
    results = []
    for i in range(top_k):
        class_name = class_names[top_idx[i]]
        probability = top_prob[i]
        results.append({
            'rank': i + 1,
            'class_name': class_name,
            'probability': float(probability),
            'confidence': f"{probability:.2%}"
        })

    # 打印结果
    print(f"\n图片文件: {os.path.basename(image_path)}")
    print(f"推理时间: {inference_time:.4f}秒")
    print(f"前{top_k}个预测结果:")
    for res in results:
        print(f"{res['rank']}. {res['class_name']} - {res['confidence']}")

    return results, inference_time


if __name__ == "__main__":
    print_banner()

    # 配置文件和权重路径
    weights_path = os.path.join(BASE_DIR, "weights", "swin_tiny_patch4_window7_224.pth")
    config_path = os.path.join(REPO_DIR, "configs", "swin", "swin_tiny_patch4_window7_224.yaml")
    class_names_path = os.path.join(BASE_DIR, "weights", "imagenet_class_index.json")

    # 检查文件是否存在
    assert os.path.exists(weights_path), f"权重文件不存在: {weights_path}"
    assert os.path.exists(config_path), f"配置文件不存在: {config_path}"
    assert os.path.exists(class_names_path), f"类别文件不存在: {class_names_path}"
    for p in TEST_IMAGES:
        assert os.path.exists(p), f"图片文件不存在: {p}"

    print("\n使用设备:", device)
    print("官方代码仓库:", REPO_DIR)
    print("权重文件:", weights_path)

    # 加载模型、配置和类别名称
    print("\n加载模型和配置...")
    model, config = load_model(weights_path, config_path)

    # 打印模型信息
    print(f"\n模型信息:")
    print(f"名称: {config['model_name']}")
    print(f"输入尺寸: {config['img_size']}x{config['img_size']}")
    print(f"窗口大小: {config['window_size']}")
    print(f"参数量: {config['param_count'] / 1e6:.2f}M")

    # 加载类别名称
    print("\n加载类别名称...")
    class_names = load_class_names(class_names_path)
    print(f"类别总数: {len(class_names)}")

    # 预测图片类别
    for image_path in TEST_IMAGES:
        print(f"\n预测图片: {image_path}")
        results, _ = predict(image_path, model, config, class_names)

    print("\n" + "-" * 70)
    print("Swin-T 图像分类完成。")
