import cv2
from ultralytics import YOLO
import os
import json
from pathlib import Path
import torch

model = None
tennis_class_ids = []


def init_model(model_path=None):
    global model, tennis_class_ids
    if model is not None:
        return

    if model_path is None:
        script_dir = Path(__file__).parent.absolute()
        model_path = script_dir / "runs" / "detect" / "train3" / "weights" / "best.pt"

    print(f"模型路径: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    try:
        print("正在加载模型...")
        model = YOLO(str(model_path))
        print("模型加载成功!")

        for idx, name in model.names.items():
            if name.lower() in ["tennis", "tennis_ball"]:
                tennis_class_ids.append(idx)

        if not tennis_class_ids:
            print("警告: 模型中没有找到网球类别! 将检测所有类别")
            tennis_class_ids = list(model.names.keys())

        print(f"将检测的类别ID: {tennis_class_ids}")

    except Exception as e:
        print(f"加载模型时出错: {e}")
        raise


def process_img(img_path, conf=0.2, save_output=True, output_dir='runs/detect'):
    global model
    if model is None:
        try:
            init_model()
        except Exception as e:
            print(f"模型初始化失败: {e}")
            return []

    if save_output:
        json_output_dir = Path(output_dir)
        image_output_dir = json_output_dir / "images"
        os.makedirs(json_output_dir, exist_ok=True)
        os.makedirs(image_output_dir, exist_ok=True)

    try:
        orig_image = cv2.imread(str(img_path))
        if orig_image is None:
            print(f"无法读取图像: {img_path}")
            return []

        results = model.predict(
            source=str(img_path),
            conf=conf,
            imgsz=(384, 640),
            verbose=False
        )

        detections = []
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    if cls_id in tennis_class_ids:
                        coords = box.xyxy.cpu().numpy().squeeze()
                        if coords.ndim == 1 and len(coords) >= 4:
                            x1, y1, x2, y2 = coords[:4]
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
                                if save_output:
                                    cv2.rectangle(orig_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                    label = f"Tennis {detection['confidence']:.2f}"
                                    cv2.putText(orig_image, label, (int(x1), int(y1)-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        detections.sort(key=lambda d: d['w'] * d['h'], reverse=True)
        output_detections = [{"x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"]} for d in detections]

        if save_output:
            img_name = Path(img_path).stem
            json_path = json_output_dir / f"{img_name}_result.json"
            try:
                with open(json_path, 'w') as f:
                    json.dump(output_detections, f, indent=4)
                print(f"JSON 结果已保存: {json_path}")
            except Exception as e:
                print(f"保存 JSON 文件时出错: {e}")

            image_output_path = image_output_dir / f"{img_name}_detected.jpg"
            try:
                cv2.imwrite(str(image_output_path), orig_image)
                print(f"带标记的图像已保存: {image_output_path}")
            except Exception as e:
                print(f"保存带标记的图像时出错: {e}")

        return output_detections

    except Exception as e:
        print(f"处理图像 {img_path} 时出错: {e}")
        return []
