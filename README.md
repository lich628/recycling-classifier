# ♻️ Recycling Classifier

This project is a machine learning-based image classifier that identifies recyclable materials such as plastic, paper, metal, etc. It uses a convolutional neural network (CNN) built with PyTorch and trained on a custom dataset of recyclable items.

---

## 📁 Project Structure

```
recycling-classifier/
│
├── datasets/         # Dataset folder (may be large, not uploaded to GitHub)
├── models/           # Trained model weights (e.g., .pth files)
├── notebooks/        # Jupyter Notebooks for training, testing, experimentation
├── test_images/      # Sample test images for prediction
├── predict.py        # Main script to run inference
├── .gitignore        # Files to exclude from Git
├── README.md         # Project description and usage guide
└── requirements.txt  # Python package dependencies
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/lich628/recycling-classifier.git
cd recycling-classifier
```

### 2. Set up virtual environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Running Predictions

To run predictions on test images using a trained model:

```bash
python predict.py
```

Make sure the model file is placed in the `models/` directory and the images are under `test_images/`.

---

## 🧠 Model

- Backbone: ResNet18 (custom final layer for classification)
- Framework: PyTorch
- Input size: 224×224 RGB images
- Loss: CrossEntropyLoss
- Optimizer: Adam

---

## 📊 Evaluation Metrics

- Accuracy
- Confusion Matrix (see notebooks)
- Sample visualizations included in notebooks

---

## 🔧 Development Tips

- Use branches to develop new features
- Use `nbstripout` to clean Jupyter notebook outputs before commits:

```bash
pip install nbstripout
nbstripout --install
```

---

## 📄 License

This project is for educational purposes under the [MIT License](LICENSE).