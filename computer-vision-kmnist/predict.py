import torch
import argparse
from PIL import Image
from torchvision import transforms


# ------------------------
# Config
# ------------------------

CLASS_NAMES = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']


# ------------------------
# Image preprocessing
# ------------------------

def load_image(image_path):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path)
    image = transform(image)

    return image


# ------------------------
# Prediction
# ------------------------

def predict(model, image_tensor):

    model.eval()

    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))
        pred_idx = torch.argmax(logits, dim=1).item()

    return pred_idx, CLASS_NAMES[pred_idx]


# ------------------------
# Main executable
# ------------------------

def main():

    parser = argparse.ArgumentParser(description="KMNIST model inference")

    parser.add_argument("--model", required=True, help="Path to model file (.pt)")
    parser.add_argument("--image", required=True, help="Path to image")

    args = parser.parse_args()

    # Load model
    model = torch.load(args.model, map_location="cpu")

    # Load image
    image_tensor = load_image(args.image)

    # Predict
    idx, label = predict(model, image_tensor)

    print("Prediction index:", idx)
    print("Prediction label:", label)


if __name__ == "__main__":
    main()
