import cv2
from ultralytics import YOLO
import os
import json
from pathlib import Path
import torch

# 全局变量
model = None
tennis_class_ids = []

def init_model(model_path=None):
    global model, tennis_class_ids
    
    # 如果已经初始化则跳过
    if model is not None:
        return
    
    # 设置默认模型路径
    if model_path is None:
        script_dir = Path(__file__).parent.absolute()
        model_path = script_dir / "runs" / "detect" / "train12" / "weights" / "best.pt"
    
    print(f"模型路径: {model_path}")
    
    # 检查模型文件是否存在
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    
    try:
        print("正在加载模型...")
        model = YOLO(str(model_path))
        print("模型加载成功!")
        
        # 确定网球类别的ID
        for idx, name in model.names.items():
            if name.lower() in ["tennis"]:
                tennis_class_ids.append(idx)
        
        if not tennis_class_ids:
            print("警告: 模型中没有找到'tennis'类别! 将检测所有类别")
            tennis_class_ids = list(model.names.keys())
        
        print(f"将检测的类别ID: {tennis_class_ids}")
        return model
    
    except Exception as e:
        print(f"加载模型时出错: {e}")
        raise

def process_img(img_path, conf=0.6, save_output=False, output_dir=None):
    """
    处理单张图片并返回检测结果
    
    参数:
        img_path: 图片路径
        conf: 置信度阈值 (默认0.6)
        save_output: 是否保存输出文件
        output_dir: 输出目录
    
    返回:
        检测结果列表 [{"x":x, "y":y, "w":w, "h":h}, ...]
    """
    global model
    
    # 确保模型已初始化
    if model is None:
        try:
            init_model()
        except Exception as e:
            print(f"模型初始化失败: {e}")
            return []
    
    # 设置输出目录
    if save_output and output_dir is None:
        script_dir = Path(__file__).parent.absolute()
        json_output_dir = script_dir / "runs" / "detect" / "predict_custom"
        image_output_dir = json_output_dir / "images"
    elif save_output:
        json_output_dir = Path(output_dir)
        image_output_dir = json_output_dir / "images"
    
    if save_output:
        os.makedirs(json_output_dir, exist_ok=True)
        os.makedirs(image_output_dir, exist_ok=True)
    
    try:
        # 读取图片
        orig_image = cv2.imread(str(img_path))
        if orig_image is None:
            print(f"无法读取图像: {img_path}")
            return []
        
        # 执行预测
        results = model.predict(
            source=str(img_path),
            conf=conf,
            imgsz=(384, 640),
            verbose=False
        )
        
        # 解析检测结果
        detections = []
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    if hasattr(box, 'cls') and box.cls is not None:
                        cls_id = int(box.cls)
                        if cls_id in tennis_class_ids:
                            if box.xyxy.numel() >= 4:
                                coords = box.xyxy.cpu().numpy().squeeze()
                                
                                # 确保坐标格式正确
                                if coords.ndim == 0 or len(coords) < 4:
                                    continue
                                    
                                if coords.ndim == 1:  # 一维数组
                                    x1, y1, x2, y2 = coords
                                else:  # 二维数组
                                    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
                                
                                w, h = x2 - x1, y2 - y1
                                if w > 0 and h > 0:
                                    detection = {
                                        "x": int(x1),
                                        "y": int(y1),
                                        "w": int(w),
                                        "h": int(h),
                                        "confidence": float(box.conf.item())
                                    }
                                    detections.append(detection)
                                    
                                    # 如果需要保存输出，绘制检测框
                                    if save_output:
                                        cv2.rectangle(orig_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                        label = f"Tennis {detection['confidence']:.2f}"
                                        cv2.putText(orig_image, label, (int(x1), int(y1)-10), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 按面积降序排序
        detections.sort(key=lambda d: d['w'] * d['h'], reverse=True)
        
        # 精简输出格式（仅坐标）
        output_detections = [{"x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"]} for d in detections]
        
        # 保存结果
        if save_output:
            img_name = Path(img_path).stem
            
            # 保存JSON结果
            json_path = json_output_dir / f"{img_name}_result.json"
            with open(json_path, 'w') as f:
                json.dump(output_detections, f, indent=4)
            
            # 保存带标记的图像
            image_output_path = image_output_dir / f"{img_name}_detected.jpg"
            cv2.imwrite(str(image_output_path), orig_image)
            print(f"结果已保存: JSON={json_path.name}, 图片={image_output_path.name}")
        
        return output_detections
    
    except Exception as e:
        print(f"处理图像 {img_path} 时出错: {e}")
        return []

# 以下为测试代码
if __name__ == '__main__':
    # 打印环境信息
    print("PyTorch版本:", torch.__version__)
    print("CUDA是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("设备名称:", torch.cuda.get_device_name(0))
    
    # 初始化模型
    try:
        init_model()
    except Exception as e:
        print(f"初始化失败: {e}")
        exit(1)
    
    # 确定测试图片目录 - 使用验证集目录
    base_dir = Path(__file__).parent.absolute()
    dataset_path = base_dir / "datasets" / "bvn" / "images" / "val"
    
    # 检查数据集路径是否存在
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
    
    print(f"找到 {len(img_paths)} 张测试图片")
    
    # 计时统计
    import time
    def now():
        return int(time.time() * 1000)
    
    last_time = 0
    count_time = 0
    max_time = 0
    min_time = float('inf')
    processed_count = 0
    
    # 处理每张测试图片
    for img_path in img_paths:
        print(f"\n[{processed_count+1}/{len(img_paths)}] 处理图片: {img_path.name}")
        
        # 计时执行
        last_time = now()
        result = process_img(img_path, conf=0.6, save_output=True)
        run_time = now() - last_time
        
        # 打印结果
        print(f"检测到 {len(result)} 个网球")
        print(f"耗时: {run_time} ms")
        
        # 更新计时统计
        processed_count += 1
        count_time += run_time
        if run_time > max_time:
            max_time = run_time
        if run_time < min_time:
            min_time = run_time
    
    # 输出统计结果
    if processed_count > 0:
        print("\n===== 性能统计 =====")
        print(f"处理图片数: {processed_count}")
        print(f"平均耗时: {int(count_time/processed_count)} ms")
        print(f"最大耗时: {max_time} ms")
        print(f"最小耗时: {min_time} ms")
    else:
        print("警告: 未处理任何图片")