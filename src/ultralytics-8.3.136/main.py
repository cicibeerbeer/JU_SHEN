import os
import time
from pathlib import Path
import argparse
from process import init_model, process_img


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('布尔值应为 true 或 false')


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="YOLO网球检测系统主程序")
    parser.add_argument("--model_path", type=str, default=None,
                        help="自定义模型路径 (默认为默认位置)")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="数据集目录 (默认为验证集)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认为predict_custom)")
    parser.add_argument("--conf_threshold", type=float, default=0.20,
                        help="置信度阈值 (默认0.20)")
    parser.add_argument("--save_output", type=str2bool, default=True,
                        help="是否保存输出文件和检测图像 (默认: 是)")

    args = parser.parse_args()

    base_dir = Path(__file__).parent.absolute()
    model_path = Path(args.model_path) if args.model_path else None
    dataset_path = Path(args.dataset_dir) if args.dataset_dir else base_dir / "datasets" / "bvn" / "images" / "val"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "runs" / "detect" / "predict_custom"

    try:
        init_model(model_path=model_path)
    except Exception as e:
        print(f"模型初始化失败: {e}")
        exit(1)

    if not dataset_path.exists():
        print(f"错误: 未找到数据集目录: {dataset_path}")
        exit(1)

    img_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        img_paths.extend(dataset_path.glob(ext))

    if not img_paths:
        print(f"警告: 在 {dataset_path} 中未找到任何图片文件!")
        exit(1)

    print(f"找到 {len(img_paths)} 张待检测图片")
    print(f"置信度阈值: {args.conf_threshold}")
    print(f"保存输出: {'是' if args.save_output else '否'}")
    if args.save_output:
        print(f"输出目录: {output_dir}")

    start_time = time.time()
    count_time = 0
    max_time = 0
    min_time = float('inf')
    processed_count = 0

    for img_path in img_paths:
        print(f"\n[{processed_count+1}/{len(img_paths)}] 处理图片: {img_path.name}")
        tic = time.time()

        result = process_img(
            img_path=img_path,
            conf=args.conf_threshold,
            save_output=args.save_output,
            output_dir=output_dir
        )
        run_time = (time.time() - tic) * 1000

        print(f"检测到 {len(result)} 个网球")
        print(f"耗时: {run_time:.2f} ms")

        processed_count += 1
        count_time += run_time
        max_time = max(max_time, run_time)
        min_time = min(min_time, run_time)

    total_time = (time.time() - start_time) * 1000
    if processed_count > 0:
        print("\n===== 性能统计 =====")
        print(f"处理图片数: {processed_count}")
        print(f"平均耗时: {count_time/processed_count:.2f} ms")
        print(f"最大耗时: {max_time:.2f} ms")
        print(f"最小耗时: {min_time:.2f} ms")
        print(f"总处理时间: {total_time:.2f} ms")
    else:
        print("警告: 未处理任何图片")


if __name__ == '__main__':
    main()
