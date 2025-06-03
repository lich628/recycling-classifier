import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Preprocessing (consistent with training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load('models/recycling_resnet18_30ep.pth', map_location=device))
model.to(device)
model.eval()

# Predict function
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = output.max(1)
        return class_names[predicted.item()]

# Upload and display
def main():
    # Use tkinter just for file dialog (cross-platform)
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if not file_path:
        print("No image selected.")
        return

    # Predict and display
    prediction = predict_image(file_path)
    print("Prediction:", prediction)

    img = Image.open(file_path).convert('RGB')
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {prediction}", fontsize=14, color='green')
    plt.show()

if __name__ == "__main__":
    main()
