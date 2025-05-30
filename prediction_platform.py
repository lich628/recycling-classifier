import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, Label

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
model.load_state_dict(torch.load('models/recycling_resnet18_20ep.pth', map_location=device))
model.to(device)
model.eval()

# Prediction
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = output.max(1)
        return class_names[predicted.item()]

# GUI logic
def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        try:
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(img)

            image_label.config(image=img_tk)
            image_label.image = img_tk  # prevent garbage collection

            result = predict_image(file_path)
            result_label.config(text=f"Prediction: {result}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

# Build GUI
root = tk.Tk()
root.title("Recyclable Material Classifier")

# GUI Layout
tk.Button(root, text="Upload Image", command=open_image, font=("Arial", 12)).pack(pady=10)
image_label = Label(root)
image_label.pack()
result_label = Label(root, text="Prediction: ", font=("Arial", 14))
result_label.pack(pady=10)

# Start GUI loop
root.mainloop()