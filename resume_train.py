"""
断点续训 (Resume) 专用脚本
用途：当意外中断时，用当前目录源码重启被中断的训练。
"""
from ultralytics import YOLO

def main():
    # 替换成您之前中断报错的那个带有 E2ELoss 的确切权重路径
    ckpt_path = "/home/mingxing/yolo_project/CrowdHuman/ultralytics/train_result/yolo26s_300e_20260509_120513/weights/last.pt"
    
    # 这里的模型加载会使用您本地修改过的 ultralytics 目录下的损失函数
    model = YOLO(ckpt_path)
    
    print(f"正在从 {ckpt_path} 恢复训练...")
    # resume=True 会自动读取之前 args.yaml 里的配置，直接续上
    model.train(resume=True)

if __name__ == "__main__":
    main()