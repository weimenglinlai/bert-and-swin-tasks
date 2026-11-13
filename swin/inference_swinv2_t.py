# -*- coding: utf-8 -*-
"""
任务二 · 任务二（2）：基于 SwinTransformerV2 的图像分类（拓展任务）

对应参考文档：
  五、实践步骤及要求 (二) 4.实践任务
    “请修改 3.代码编写和运行 小节中的代码，使用 SwinTransformerV2
      （SwinTransformerV2 是 Swin-Transformer 的改进版本）的预训练模型
      swinv2_tiny_patch4_window8_256.pth 进行图像分类。”

相对 Swin-T 版本，本脚本做了以下对应调整：
  1. 导入的类：from models.swin_transformer_v2 import SwinTransformerV2
  2. 配置文件：configs/swinv2/swinv2_tiny_patch4_window8_256.yaml
     —— 该文件的结构与 Swin-T 不同：图像尺寸写在 DATA.IMG_SIZE，
        模型超参放在 MODEL.SWINV2 下，因此读取方式需相应修改；
  3. 权重文件：swinv2_tiny_patch4_window8_256.pth
  4. 输入尺寸由 224 变为 256，窗口大小由 7 变为 8；
  5. SwinTransformerV2 类新增 pretrained_window_sizes 参数（默认 [0,0,0,0]），
     本次使用 ImageNet-1K 单阶段微调权重，取默认值即可。
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
sys.path.insert(0, REPO_DIR)

# 屏蔽与推理无关的告警（如 torch.meshgrid 的弃用提示），让控制台输出更干净
warnings.filterwarnings("ignore")

# 修改点 1：从 models.swin_transformer_v2 导入 SwinTransformerV2 类
from models.swin_transformer_v2 import SwinTransformerV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    print("子任务：基于 SwinTransformerV2 的图像分类（SwinV2-T 模型）")
    print("=" * 70)


def load_model(weights_path, config_path):
    """
    加载 SwinTransformerV2 模型与预训练权重
    参数:
        weights_path (str): 预训练权重文件路径
        config_path (str): 模型配置文件路径
    返回:
        torch.nn.Module: 加载权重后的模型
        dict: 模型配置信息
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config['MODEL']
    # 修改点 2：SwinV2 的配置文件把超参放在 SWINV2 节点下
    swinv2_cfg = model_cfg['SWINV2']
    # 修改点 3：SwinV2 配置里图像尺寸写在 DATA.IMG_SIZE（不再从模型名里解析）
    img_size = int(config['DATA']['IMG_SIZE'])

    # 修改点 4：初始化的是 SwinTransformerV2，并额外传入 pretrained_window_sizes
    model = SwinTransformerV2(
        img_size=img_size,               # 256
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=swinv2_cfg['EMBED_DIM'],      # 96
        depths=swinv2_cfg['DEPTHS'],            # [2, 2, 6, 2]
        num_heads=swinv2_cfg['NUM_HEADS'],      # [3, 6, 12, 24]
        window_size=swinv2_cfg['WINDOW_SIZE'],  # 8
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=model_cfg['DROP_PATH_RATE'],  # 0.2
        norm_layer=torch.nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        pretrained_window_sizes=[0, 0, 0, 0]   # ImageNet-1K 单阶段权重取默认值
    )

    checkpoint = torch.load(weights_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model, {
        'img_size': img_size,
        'model_name': model_cfg['NAME'],
        'window_size': swinv2_cfg['WINDOW_SIZE'],
        'embed_dim': swinv2_cfg['EMBED_DIM'],
        'depths': swinv2_cfg['DEPTHS'],
        'param_count': param_count,
    }


def load_class_names(json_path):
    """加载ImageNet类别名称映射"""
    with open(json_path, 'r') as f:
        class_idx = json.load(f)
    return {int(k): v[1] for k, v in class_idx.items()}


def preprocess_image(image_path, img_size):
    """根据模型输入尺寸预处理图像（SwinV2-T 输入为 256×256）"""
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
    image = preprocess_image(image_path, config['img_size'])

    start_time = time.time()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    inference_time = time.time() - start_time

    top_prob, top_idx = torch.topk(probabilities, k=top_k)
    top_prob = top_prob.squeeze().cpu().numpy()
    top_idx = top_idx.squeeze().cpu().numpy()

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

    print(f"\n图片文件: {os.path.basename(image_path)}")
    print(f"推理时间: {inference_time:.4f}秒")
    print(f"前{top_k}个预测结果:")
    for res in results:
        print(f"{res['rank']}. {res['class_name']} - {res['confidence']}")

    return results, inference_time


if __name__ == "__main__":
    print_banner()

    # 修改点 5：权重文件与配置文件都换成 SwinV2-T 的
    weights_path = os.path.join(BASE_DIR, "weights", "swinv2_tiny_patch4_window8_256.pth")
    config_path = os.path.join(REPO_DIR, "configs", "swinv2", "swinv2_tiny_patch4_window8_256.yaml")
    class_names_path = os.path.join(BASE_DIR, "weights", "imagenet_class_index.json")

    assert os.path.exists(weights_path), f"权重文件不存在: {weights_path}"
    assert os.path.exists(config_path), f"配置文件不存在: {config_path}"
    assert os.path.exists(class_names_path), f"类别文件不存在: {class_names_path}"
    for p in TEST_IMAGES:
        assert os.path.exists(p), f"图片文件不存在: {p}"

    print("\n使用设备:", device)
    print("官方代码仓库:", REPO_DIR)
    print("权重文件:", weights_path)

    print("\n加载模型和配置...")
    model, config = load_model(weights_path, config_path)

    print(f"\n模型信息:")
    print(f"名称: {config['model_name']}")
    print(f"输入尺寸: {config['img_size']}x{config['img_size']}")
    print(f"窗口大小: {config['window_size']}")
    print(f"嵌入维度: {config['embed_dim']}    各阶段深度: {config['depths']}")
    print(f"参数量: {config['param_count'] / 1e6:.2f}M")

    print("\n加载类别名称...")
    class_names = load_class_names(class_names_path)
    print(f"类别总数: {len(class_names)}")

    for image_path in TEST_IMAGES:
        print(f"\n预测图片: {image_path}")
        results, _ = predict(image_path, model, config, class_names)

    print("\n" + "-" * 70)
    print("SwinV2-T 图像分类完成。")
