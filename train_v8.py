import argparse
import os
import subprocess

def train_yolov8(data, cfg, epochs, batch_size, img_size, device, weights):
    command = [
        'python', 'D:\\ultralytics-main-20241022\\ultralytics\\train_v8.py',  # 假设 train.py 是 YOLOv8 的训练脚本
        '--data', data,
        '--cfg', cfg,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--img', str(img_size),
        '--device', device,
        '--weights', weights,  # 使用预训练权重
    ]

    print("开始训练...")
    subprocess.run(command)
    print("训练完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 训练脚本")
    parser.add_argument('--data', type=str, required=True, help='数据集配置文件路径')
    parser.add_argument('--cfg', type=str, required=True, help='模型配置文件路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='每批次样本数')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像大小')
    parser.add_argument('--device', type=str, default='0', help='使用的 GPU 设备')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='预训练权重路径')

    args = parser.parse_args()

    train_yolov8(args.data, args.cfg, args.epochs, args.batch_size, args.img_size, args.device, args.weights)
