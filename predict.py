import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import sys

# 测试pull request wty

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义类别顺序（需与 ImageFolder 加载顺序一致）
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 图像预处理方法（需与训练时一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 加载模型
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load('models/recycling_resnet18_20ep.pth', map_location=device))
model.to(device)
model.eval()

# 预测函数
def predict(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"无法读取图像: {e}")
        return

    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = output.max(1)
        print(f"\n图像 {img_path} 预测结果：{class_names[predicted.item()]}\n")

# 通过命令行参数指定图像路径
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python predict.py path/to/image.jpg")
    else:
        predict(sys.argv[1])
