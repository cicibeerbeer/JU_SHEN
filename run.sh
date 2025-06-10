#!/bin/bash

# run.sh - YOLO网球检测系统启动器

# 设置保存输出选项 (true 或 false)
SAVE_OUTPUT=true

# 设置项目根目录
PROJECT_ROOT="$HOME/桌面/JU_SHEN"
cd "$PROJECT_ROOT" || { echo "错误: 无法进入项目目录"; exit 1; }
echo "工作目录: $PWD"

# 设置关键路径
YOLO_DIR="./src/ultralytics-8.3.136"
MODEL_PATH="$YOLO_DIR/runs/detect/train9/weights/best.pt"
VALIDATION_DIR="./src/ultralytics-8.3.136/datasets/bvn/images/val"
PYTHON_SCRIPT="$YOLO_DIR/main.py"
OUTPUT_DIR="$YOLO_DIR/runs/detect/predict_custom"

# 验证文件路径
echo "验证系统路径..."
[ ! -f "$MODEL_PATH" ] && { echo "错误: 未找到模型文件 $MODEL_PATH"; exit 1; }
[ ! -d "$VALIDATION_DIR" ] && { echo "错误: 未找到验证集 $VALIDATION_DIR"; exit 1; }
[ ! -f "$PYTHON_SCRIPT" ] && { echo "错误: 未找到Python脚本 $PYTHON_SCRIPT"; exit 1; }

# 确保python可执行
PYTHON_EXEC=$(which python)
[ -z "$PYTHON_EXEC" ] && { echo "错误: 未找到Python解释器"; exit 1; }

# 创建输出目录
mkdir -p "$OUTPUT_DIR" 2>/dev/null

# 启动参数
CONFIDENCE=0.45

echo "==============================================="
echo "网球检测系统启动"
echo "模型: $MODEL_PATH"
echo "验证集: $VALIDATION_DIR (包含 $(ls "$VALIDATION_DIR" | wc -l) 张图片)"
echo "输出目录: $OUTPUT_DIR"
echo "置信度阈值: $CONFIDENCE"
echo "保存输出: $SAVE_OUTPUT"
echo "开始时间: $(date +'%Y-%m-%d %H:%M:%S')"
echo "==============================================="

# 执行网球检测
start_time=$(date +%s)

# 构建参数
PYTHON_ARGS=(
    "--model_path" "$MODEL_PATH"
    "--dataset_dir" "$VALIDATION_DIR"
    "--output_dir" "$OUTPUT_DIR"
    "--conf_threshold" "$CONFIDENCE"
    "--save_output" "$SAVE_OUTPUT"
)

python "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}"
exit_code=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

# 显示结果摘要
if [ $exit_code -eq 0 ]; then
    echo "==============================================="
    echo "网球检测完成!"
    echo "处理时间: $duration 秒"
    
    if [ "$SAVE_OUTPUT" = true ]; then
        echo "输出文件:"
        echo "  - JSON文件: $(find "$OUTPUT_DIR" -type f -name "*_result.json" | wc -l) 个"
        echo "  - 标注图片: $(find "$OUTPUT_DIR/images" -type f -name "*_detected.jpg" | wc -l) 张"
        
        # 显示前5个输出文件作为示例
        echo "示例输出文件:"
        find "$OUTPUT_DIR" -maxdepth 1 -type f -name "*_result.json" | head -n 5
    fi
    
    echo "结束时间: $(date +'%Y-%m-%d %H:%M:%S')"
    echo "==============================================="
else
    echo "错误: 检测过程中出现错误 (退出码: $exit_code)"
    exit $exit_code
fi