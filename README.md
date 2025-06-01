训练模型
yolo detect train model=yolov8n.pt data=yolobvn.yaml epochs=100 imgsz=640
验证模型  模型用你的训练产生的trainx
yolo val model=./runs/detect/train10/weights/best.pt data=yolobvn.yaml
yolo val model=./runs/detect/train10/weights/best.pt data=yolobvn.yaml split=train  验证训练集
  
预测
yolo predict model=./runs/detect/train10/weights/best.pt source=./datasets/bvn/images/train show=True   这里是你放图片的地址
