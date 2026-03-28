import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import FeatureExtractor
def infer_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
# Load the trained model from a checkpoint

def load_model(model_path, embedding_dimension, device):
    model = FeatureExtractor(embedding_dim=embedding_dimension).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model
# Inference function to get embedding for a single image
@torch.no_grad()
def infer_image(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    transform = infer_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)
    embedding = model(image_tensor)
    return embedding.cpu()
@torch.no_grad()
def get_embedding_for_multiple_images(model, image_paths, device):
    images = []
    transform = infer_transform()
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image = transform(image)
        images.append(image)
    batch = torch.stack(images).to(device)
    embeddings = model(batch)
    return embeddings.cpu()
# Load embeddings and labels from a .pt file
def main():
    parser = argparse.ArgumentParser(description="Inference script for image embeddings")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--image_paths", type=str, nargs="+", required=True, help="One or more image paths")
    parser.add_argument("--embedding_dimension", type=int, default=128, help="Dimension of the embedding space")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(args.model_path, args.embedding_dimension, device)
    embeddings = get_embedding_for_multiple_images(model, args.image_paths, device)

    for path, embedding in zip(args.image_paths, embeddings):
     print(f"\nimage: {path}")
     print(f"Embedding shape: {embedding.shape}")
     print(embedding.numpy())
if __name__ == "__main__":   
    main()
