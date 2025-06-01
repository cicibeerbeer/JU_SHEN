import os

image_dir = './datasets/bvn/images/train'
label_dir = './datasets/bvn/labels/train'

image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

for image_file in image_files:
    label_file = image_file.replace('.jpg', '.txt')
    label_path = os.path.join(label_dir, label_file)

    if not os.path.exists(label_path):
        # 创建一个空标签文件
        open(label_path, 'w').close()
        print(f'创建空标签文件: {label_path}')
