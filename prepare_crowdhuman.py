"""
CrowdHuman 数据集准备脚本
功能：
  1. 解压图片到 images/train, images/val
  2. 将 ODGT 标注转换为 YOLO 格式 (labels/train, labels/val)
  3. 生成 crowdhuman.yaml 数据集配置.
"""

import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

# ======================== 配置 ========================
DATASET_DIR = Path("dataset_CrowdHuman")
OUTPUT_DIR = Path("dataset_CrowdHuman_yolo")

TRAIN_ZIPS = [
    DATASET_DIR / "CrowdHuman_train01.zip",
    DATASET_DIR / "CrowdHuman_train02.zip",
    DATASET_DIR / "CrowdHuman_train03.zip",
]
VAL_ZIP = DATASET_DIR / "CrowdHuman_val.zip"
TRAIN_ANNOTATION = DATASET_DIR / "annotation_train.odgt.txt"
VAL_ANNOTATION = DATASET_DIR / "annotation_val.odgt.txt"

# 只检测 person 类
CLASS_NAMES = ["person"]
NUM_CLASSES = len(CLASS_NAMES)


def extract_images(zip_paths, output_dir):
    """从 zip 文件中解压图片."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    for zip_path in zip_paths:
        print(f"  解压 {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            for info in z.infolist():
                # 路径格式: Images/xxx.jpg -> 提取文件名
                if not info.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                filename = os.path.basename(info.filename)
                target = output_dir / filename
                if target.exists():
                    extracted += 1
                    continue
                # 从 zip 中读取并写出
                with z.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                extracted += 1
        print(f"    累计解压 {extracted} 张")
    return extracted


def parse_odgt(annotation_file):
    """解析 ODGT 标注文件."""
    samples = []
    with open(annotation_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            samples.append(data)
    return samples


def get_image_size(image_path):
    """获取图片尺寸."""
    with Image.open(image_path) as img:
        return img.width, img.height


def convert_annotation(sample, img_dir, label_dir):
    """将单个 ODGT 标注转换为 YOLO 格式."""
    image_id = sample["ID"]
    gtboxes = sample.get("gtboxes", [])

    # 查找对应图片
    img_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = img_dir / f"{image_id}{ext}"
        if candidate.exists():
            img_path = candidate
            break

    if img_path is None:
        return None

    # 获取图片尺寸
    img_w, img_h = get_image_size(img_path)

    yolo_lines = []
    person_count = 0
    skip_count = 0

    for box in gtboxes:
        # 只保留 person 标签
        if box.get("tag") != "person":
            skip_count += 1
            continue

        # 跳过被标记为 ignore 的样本
        extra = box.get("extra", {})
        head_attr = box.get("head_attr", {})

        if extra.get("ignore", 0) == 1:
            skip_count += 1
            continue
        if head_attr.get("ignore", 0) == 1:
            skip_count += 1
            continue

        # 使用 fbox (全身框) 作为检测目标
        fbox = box.get("fbox")
        if fbox is None:
            skip_count += 1
            continue

        x, y, w, h = fbox

        # 过滤无效框
        if w <= 0 or h <= 0:
            skip_count += 1
            continue

        # 裁剪到图片边界内
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        w = x2 - x
        h = y2 - y

        if w <= 0 or h <= 0:
            skip_count += 1
            continue

        # 转换为 YOLO 格式: class x_center y_center width height (归一化)
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h

        # 确保值在 [0, 1] 范围内
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        norm_w = max(0, min(1, norm_w))
        norm_h = max(0, min(1, norm_h))

        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
        person_count += 1

    # 写入标签文件
    label_path = label_dir / f"{image_id}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines) + "\n" if yolo_lines else "")

    return {
        "image_id": image_id,
        "persons": person_count,
        "skipped": skip_count,
        "has_person": person_count > 0,
    }


def create_dataset_yaml(output_dir):
    """生成 CrowdHuman 数据集 YAML 配置."""
    yaml_content = f"""# CrowdHuman Dataset
# 密集人群行人检测数据集
# 使用 fbox (全身框) 作为检测目标

path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes
nc: {NUM_CLASSES}
names: {CLASS_NAMES}
"""
    yaml_path = output_dir / "crowdhuman.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"  数据集配置已保存: {yaml_path}")
    return yaml_path


def main():
    print("=" * 60)
    print("CrowdHuman 数据集准备")
    print("=" * 60)

    # 创建输出目录
    dirs = {
        "train_img": OUTPUT_DIR / "images" / "train",
        "val_img": OUTPUT_DIR / "images" / "val",
        "train_lbl": OUTPUT_DIR / "labels" / "train",
        "val_lbl": OUTPUT_DIR / "labels" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: 解压图片
    print("\n[Step 1/4] 解压训练集图片...")
    train_count = extract_images(TRAIN_ZIPS, dirs["train_img"])
    print(f"  训练集图片: {train_count} 张")

    print("\n[Step 1/4] 解压验证集图片...")
    val_count = extract_images([VAL_ZIP], dirs["val_img"])
    print(f"  验证集图片: {val_count} 张")

    # Step 2: 转换训练集标注
    print("\n[Step 2/4] 转换训练集标注...")
    train_samples = parse_odgt(TRAIN_ANNOTATION)
    print(f"  训练集标注条目: {len(train_samples)}")

    train_stats = defaultdict(int)
    for i, sample in enumerate(train_samples):
        result = convert_annotation(sample, dirs["train_img"], dirs["train_lbl"])
        if result:
            train_stats["converted"] += 1
            train_stats["total_persons"] += result["persons"]
            train_stats["total_skipped"] += result["skipped"]
            if result["has_person"]:
                train_stats["images_with_person"] += 1
        if (i + 1) % 2000 == 0:
            print(f"    已处理 {i + 1}/{len(train_samples)}")

    print("  训练集转换完成:")
    print(f"    转换图片数: {train_stats['converted']}")
    print(f"    含行人图片: {train_stats['images_with_person']}")
    print(f"    总行人框数: {train_stats['total_persons']}")
    print(f"    跳过标注数: {train_stats['total_skipped']}")

    # Step 3: 转换验证集标注
    print("\n[Step 3/4] 转换验证集标注...")
    val_samples = parse_odgt(VAL_ANNOTATION)
    print(f"  验证集标注条目: {len(val_samples)}")

    val_stats = defaultdict(int)
    for i, sample in enumerate(val_samples):
        result = convert_annotation(sample, dirs["val_img"], dirs["val_lbl"])
        if result:
            val_stats["converted"] += 1
            val_stats["total_persons"] += result["persons"]
            val_stats["total_skipped"] += result["skipped"]
            if result["has_person"]:
                val_stats["images_with_person"] += 1
        if (i + 1) % 1000 == 0:
            print(f"    已处理 {i + 1}/{len(val_samples)}")

    print("  验证集转换完成:")
    print(f"    转换图片数: {val_stats['converted']}")
    print(f"    含行人图片: {val_stats['images_with_person']}")
    print(f"    总行人框数: {val_stats['total_persons']}")
    print(f"    跳过标注数: {val_stats['total_skipped']}")

    # Step 4: 生成数据集配置
    print("\n[Step 4/4] 生成数据集配置...")
    yaml_path = create_dataset_yaml(OUTPUT_DIR)

    # 汇总
    print("\n" + "=" * 60)
    print("数据准备完成!")
    print("=" * 60)
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  训练集: {train_count} 张图片, {train_stats['total_persons']} 个行人标注")
    print(f"  验证集: {val_count} 张图片, {val_stats['total_persons']} 个行人标注")
    print(f"  数据集配置: {yaml_path}")
    print("\n训练命令:")
    print(
        f"  conda run -n cu128 python train.py --data {yaml_path} --cfg ultralytics/cfg/models/11/yolo11s.yaml --weights yolo11s.pt --epochs 100 --batch 8 --img 640 --device 0"
    )


if __name__ == "__main__":
    main()
