# main.py - YOLO网球检测系统主程序

import os
import time
from pathlib import Path
import argparse
from process import init_model, process_img  # 导入process模块中的函数

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="YOLO网球检测系统主程序")
    parser.add_argument("--model_path", type=str, default=None,
                        help="自定义模型路径 (默认为默认位置)")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="数据集目录 (默认为验证集)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认为predict_custom)")
    parser.add_argument("--conf_threshold", type=float, default=0.6,
                        help="置信度阈值 (默认0.6)")
    parser.add_argument("--save_output", action='store_true',
                        help="是否保存输出文件和检测图像")
    
    args = parser.parse_args()
    
    # 获取基础路径
    base_dir = Path(__file__).parent.absolute()
    
    # 设置模型路径
    model_path = Path(args.model_path) if args.model_path else None
    
    # 设置数据集路径
    if args.dataset_dir:
        dataset_path = Path(args.dataset_dir)
    else:
        # 默认验证集位置
        dataset_path = base_dir / "datasets" / "bvn" / "images" / "val"
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir / "runs" / "detect" / "predict_custom"
    
    # 初始化模型
    try:
        init_model(model_path=model_path)
    except Exception as e:
        print(f"模型初始化失败: {e}")
        exit(1)
    
    # 确保数据集目录存在
    if not dataset_path.exists():
        print(f"错误: 未找到数据集目录: {dataset_path}")
        exit(1)
    
    # 获取所有图片文件
    img_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        img_paths.extend(dataset_path.glob(ext))
    
    # 检查是否有图片
    if not img_paths:
        print(f"警告: 在 {dataset_path} 中未找到任何图片文件!")
        exit(1)
    
    print(f"找到 {len(img_paths)} 张待检测图片")
    print(f"置信度阈值: {args.conf_threshold}")
    print(f"保存输出: {'是' if args.save_output else '否'}")
    if args.save_output:
        print(f"输出目录: {output_dir}")
    
    # 计时统计
    start_time = time.time()
    count_time = 0
    max_time = 0
    min_time = float('inf')
    processed_count = 0
    
    # 处理每张测试图片
    for img_path in img_paths:
        print(f"\n[{processed_count+1}/{len(img_paths)}] 处理图片: {img_path.name}")
        
        # 计时执行
        tic = time.time()
        result = process_img(
            img_path=img_path, 
            conf=args.conf_threshold, 
            save_output=args.save_output,
            output_dir=output_dir
        )
        run_time = (time.time() - tic) * 1000  # 毫秒
        
        # 打印结果
        print(f"检测到 {len(result)} 个网球")
        print(f"耗时: {run_time:.2f} ms")
        
        # 更新计时统计
        processed_count += 1
        count_time += run_time
        if run_time > max_time:
            max_time = run_time
        if run_time < min_time:
            min_time = run_time
    
    # 输出统计结果
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