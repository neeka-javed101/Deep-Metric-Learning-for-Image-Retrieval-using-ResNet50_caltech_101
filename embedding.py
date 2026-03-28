import torch
import os
# Utility functions for saving/loading embeddings and labels
def save_embeddings(path, embeddings, labels, image_paths=None):
    if not path.endswith(".pt"):
        path += ".pt"
    data={
       "embeddings": embeddings.detach().cpu(),
       "labels": labels.detach().cpu(),
    }
    if image_paths is not None:
        data["image_paths"]=image_paths
    torch.save(data,path)
    print(f"Embeddings saved to {path}")
# Load embeddings and labels from a .pt file
def load_embeddings(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data=torch.load(path,map_location=torch.device('cpu'))
    embeddings=data["embeddings"]
    labels=data["labels"]
    image_paths=data.get("image_paths",None)
    print(f"Embeddings loaded from {path}")
    return embeddings, labels, image_paths
   