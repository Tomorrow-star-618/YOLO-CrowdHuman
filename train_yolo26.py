"""
YOLO26 基线训练脚本 - CrowdHuman 密集人群行人检测

用法:
  conda run -n cu128 python train_baseline.py --scale s
  conda run -n cu128 python train_baseline.py --scale m --epochs 50 --batch 4
  conda run -n cu128 python train_baseline.py --scale n --epochs 100 --batch 16 --imgsz 640
"""

import argparse
import json
import os
import time

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26 基线训练 - CrowdHuman")
    parser.add_argument("--scale", type=str, default="s", choices=["n", "s", "m", "l", "x"],
                        help="模型尺度: n/nano, s/small, m/medium, l/large, x/xlarge (默认: s)")
    parser.add_argument("--weights", type=str, default=None, 
                        help="自定义预训练权重路径(如果要继承中断的last.pt，填入该路径。默认留空则使用官方原始权重)")
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数 (默认: 300)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (默认: 根据尺度自动设置)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸 (默认: 640)")
    parser.add_argument("--device", type=int, default=0, help="GPU设备号 (默认: 0)")
    parser.add_argument("--workers", type=int, default=2, help="数据加载线程数 (为WSL体验优化，默认调至: 2)")
    parser.add_argument("--patience", type=int, default=50, help="早停耐心值 (默认: 50，适配300轮需适当调大)")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率 (默认: 0.01)")
    parser.add_argument("--save-period", type=int, default=30, help="每隔N个epoch保存权重 (默认: 30)")
    return parser.parse_args()


# 各尺度推荐 batch size (针对边办公边跑的 WSL 适当保守，释放显存)
DEFAULT_BATCH = {
    "n": 8,
    "s": 8,
    "m": 4,
    "l": 2,
    "x": 2,
}


def main():
    args = parse_args()

    # 模型与路径配置
    model_name = f"yolo26{args.scale}"
    model_file = args.weights if args.weights else f"{model_name}.pt"
    batch = args.batch if args.batch else DEFAULT_BATCH[args.scale]
    
    # 学术标准化的命名与路径
    run_time_str = time.strftime("%Y%m%d_%H%M%S")
    project_dir = os.path.abspath("train_result")  # 使用绝对路径，防止 YOLO 自动套娃到 runs/detect 下面
    run_name = f"{model_name}_{args.epochs}e_{run_time_str}"

    print("=" * 60)
    print(f"YOLO26{args.scale} 期刊基线训练 - CrowdHuman")
    print("=" * 60)
    print(f"模型:      {model_file}")
    print(f"数据集:    dataset_CrowdHuman_yolo/crowdhuman.yaml")
    print(f"Epochs:    {args.epochs}")
    print(f"Batch:     {batch}")
    print(f"Image:     {args.imgsz}")
    print(f"Device:    cuda:{args.device}")
    print(f"LR0:       {args.lr0}")
    print(f"Patience:  {args.patience}")
    print(f"保存路径:  {project_dir}/{run_name}")
    print("=" * 60)

    start_time = time.time()

    # 加载预训练模型
    model = YOLO(model_file)

    # 训练 (含期刊级别控制变量)
    results = model.train(
        data="dataset_CrowdHuman_yolo/crowdhuman.yaml",
        epochs=args.epochs,
        batch=batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        patience=args.patience,
        save=True,
        save_period=args.save_period,
        plots=True,
        verbose=True,
        lr0=args.lr0,
        
        # --- 学术期刊基线实验配置 (Ablation Study 控制变量) ---
        deterministic=True,  # 确保实验的可重复性，固定随机算法和种子
        amp=True,            # 自动混合精度加速 (若后续魔改出现 nan，可设为 False)
        val=True,            # 强制开启验证，并在每个epoch后评估模型表现
        optimizer='auto',    # 显式跟踪优化器(auto下通常300轮自动用SGD)
        mosaic=1.0,          # 显式控制马赛克增强开启概率
        mixup=0.0,           # 显式关闭Mixup增强(密集人群极易严重干扰)
        copy_paste=0.0,      # 关闭额外遮挡粘贴增强控制基准纯粹
        
        # --- 为 WSL 及非专用服务器办公妥协的资源控制配置 ---
        cache=False,         # 强制关闭图片全量加载到内存(防止WSL内存爆炸引发卡顿，默认即False，在此显式固定)
    )

    elapsed = time.time() - start_time

    # 读取最终指标
    results_csv = f"{project_dir}/{run_name}/results.csv"
    final_metrics = {}
    if os.path.exists(results_csv):
        with open(results_csv) as f:
            lines = f.readlines()
            if len(lines) > 1:
                headers = [h.strip() for h in lines[0].split(",")]
                values = [v.strip() for v in lines[-1].split(",")]
                for h, v in zip(headers, values):
                    try:
                        final_metrics[h] = float(v)
                    except ValueError:
                        final_metrics[h] = v

    # 保存训练摘要 JSON (为了代码可读)
    summary = {
        "model": model_name,
        "run_name": run_name,
        "epochs": args.epochs,
        "batch_size": batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "patience": args.patience,
        "total_time_hours": round(elapsed / 3600, 2),
        "final_metrics": final_metrics,
    }
    summary_path = f"{project_dir}/{run_name}/train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ==========================================
    # 额外补充的 TXT 可读复盘日志 (学术手账)
    # ==========================================
    txt_log_path = f"{project_dir}/{run_name}/experiment_record_{run_time_str}.txt"
    with open(txt_log_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"【学术实验复盘记录表】 - 密集人群目标检测\n")
        f.write(f"实验命名: {run_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(">>> [ 一、 基础参数设定 ] <<<\n")
        f.write(f"使用框架和模型: {model_name}\n")
        f.write(f"目标轮数(Epochs): {args.epochs}\n")
        f.write(f"批处理大小(Batch): {batch}\n")
        f.write(f"图像输入尺寸: {args.imgsz}\n")
        f.write(f"初始学习率(LR): {args.lr0}\n")
        f.write(f"早停等待轮数: {args.patience}\n\n")
        
        f.write(">>> [ 二、 学术期刊控制变量声明 (Ablation Control) ] <<<\n")
        f.write("1. 随机种子 (Deterministic): True (已固定，保证复现性)\n")
        f.write("2. 验证开关 (Val): True (在每轮或按期验证评估)\n")
        f.write("3. 精度加速 (AMP): True (使用FP16混合精度)\n")
        f.write("4. 优化算法 (Optimizer): Default/Auto (通常长周期为SGD)\n")
        f.write("5. 马赛克增强 (Mosaic): 1.0 (全概率开启)\n")
        f.write("6. 图像混合 (Mixup) : 0.0 (刻意关闭，避免密集拥挤场景重叠穿模)\n")
        f.write("7. 复制粘贴 (CopyPaste): 0.0 (关闭以保证场景原生性)\n\n")
        
        f.write(">>> [ 三、 训练物理耗时与结果 ] <<<\n")
        f.write(f"总耗时: {elapsed/3600:.2f} 小时\n")
        f.write("最终验证集指标 (从最后或最佳一轮截取):\n")
        for k, v in final_metrics.items():
            f.write(f"  - {k}: {v}\n")
        f.write("\n【个人复盘备注(在此写下关于这组实验的改进点)】：\n")
        f.write("1. \n2. \n")

    print("\n" + "=" * 60)
    print("周期刊规格试验部署完毕，代码运行完成!")
    print(f"总耗时: {elapsed/3600:.2f} 小时")
    print(f"实验结果均已保存在 -> {project_dir}/{run_name}")
    print(f"实验回溯文稿路径 -> {txt_log_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
