# YOLO-CrowdHuman

基于 YOLO11 的密集人群行人检测改进方法，针对 CrowdHuman 数据集中的遮挡问题进行优化。

## 研究背景

密集人群场景下的行人检测是计算机视觉领域的重要课题。在地铁站、体育场馆、大型活动等场景中，行人之间存在严重遮挡，导致传统检测器出现大量漏检和误检。本项目基于 YOLO11 框架，针对密集遮挡场景进行改进，主要贡献包括：

- **可见区域感知训练**：利用 CrowdHuman 数据集独有的 vbox（可见身体区域）标注，设计双标签监督策略，使网络在遮挡条件下学习关注未被遮挡的部分
- **遮挡感知 NMS**：改进后处理策略，根据 IoU 和可见比例动态调整抑制阈值，降低遮挡场景下的误抑制
- **注意力增强特征融合**：在 Neck 部分引入 CBAM 注意力模块，增强模型对关键区域的特征表达能力

## 环境配置

```bash
# 创建 conda 环境
conda create -n cu128 python=3.10
conda activate cu128

# 安装 PyTorch (CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 安装 ultralytics
pip install ultralytics

# 安装其他依赖
pip install pillow opencv-python
```

## 数据集准备

### 1. 下载 CrowdHuman 数据集

从 [CrowdHuman 官网](https://www.crowdhuman.org/) 下载数据集，将以下文件放入 `dataset_CrowdHuman/` 目录：

```
dataset_CrowdHuman/
├── CrowdHuman_train01.zip
├── CrowdHuman_train02.zip
├── CrowdHuman_train03.zip
├── CrowdHuman_val.zip
├── annotation_train.odgt.txt
└── annotation_val.odgt.txt
```

### 2. 数据格式转换

运行数据准备脚本，将原始 ODGT 标注转换为 YOLO 格式：

```bash
python prepare_crowdhuman.py
```

转换完成后生成 `dataset_CrowdHuman_yolo/` 目录：

```
dataset_CrowdHuman_yolo/
├── crowdhuman.yaml          # 数据集配置
├── images/
│   ├── train/               # 15,000 张训练图片
│   └── val/                 # 4,370 张验证图片
└── labels/
    ├── train/               # 训练标签
    └── val/                 # 验证标签
```

## 数据集统计

| 子集 | 图片数 | 行人标注框数 | 平均每图人数 |
|------|--------|-------------|-------------|
| 训练集 | 15,000 | 332,805 | 22.2 |
| 验证集 | 4,370 | 97,604 | 22.3 |

- 超过 20 人的图片占比 36%
- 超过 50 人的图片占比 7.6%
- 最大拥挤度：单张图片 374 人

## 模型训练

### 基线模型训练

```bash
# YOLO11s 基线
yolo detect train data=dataset_CrowdHuman_yolo/crowdhuman.yaml \
    cfg=ultralytics/cfg/models/11/yolo11s.yaml \
    weights=yolo11s.pt \
    epochs=100 batch=8 imgsz=640 device=0
```

### 改进模型训练

```bash
# 改进后的模型（待实现）
yolo detect train data=dataset_CrowdHuman_yolo/crowdhuman.yaml \
    cfg=ultralytics/cfg/models/11/yolo11s_crowd.yaml \
    weights=yolo11s.pt \
    epochs=100 batch=8 imgsz=640 device=0
```

## 项目结构

```
YOLO-CrowdHuman/
├── README.md
├── prepare_crowdhuman.py          # 数据准备脚本
├── dataset_CrowdHuman/            # 原始数据 (git ignored)
├── dataset_CrowdHuman_yolo/       # YOLO格式数据 (git ignored)
├── ultralytics/
│   ├── cfg/
│   │   ├── models/11/             # YOLO11 模型配置
│   │   └── default.yaml           # 默认训练参数
│   ├── nn/
│   │   ├── modules/               # 网络模块定义
│   │   └── tasks.py               # 模型构建
│   ├── data/                      # 数据加载
│   ├── engine/                    # 训练引擎
│   └── utils/
│       └── loss.py                # 损失函数
└── runs/                          # 训练结果 (git ignored)
```

## 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [CrowdHuman Dataset](https://www.crowdhuman.org/)
