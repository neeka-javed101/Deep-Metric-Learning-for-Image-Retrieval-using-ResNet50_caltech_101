import os
import random
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
@torch.no_grad()
def get_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    all_images = []
    for images, labels in dataloader:
        images = images.to(device)
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())
        all_images.append(images.cpu())
    all_embeddings = torch.cat(all_embeddings,dim=0)
    all_labels = torch.cat(all_labels,dim=0)
    all_images = torch.cat(all_images,dim=0)
    return all_embeddings, all_labels, all_images
def recall_at_k(embeddings, labels, k=1):
    similarity_matrix = embeddings @ embeddings.t()
    n = embeddings.size(0)
    k = min(k, n - 1)

    recall = 0.0
    for i in range(n):
        scores = similarity_matrix[i].clone()
        scores[i] = -1e9
        top_k_indices = torch.topk(scores, k=k).indices
        if (labels[top_k_indices] == labels[i]).any():
            recall += 1.0

    return recall / n
    
def compute_recall_at_k(embeddings, labels):
    recall1 = recall_at_k(embeddings, labels, k=1)
    recall5 = recall_at_k(embeddings, labels, k=5)
    
    return recall1, recall5
    
  # Function to plot t-SNE visualization of embeddings with labels as colors 
def plot_tsne(embeddings, labels, save_path="tsne.png", title="t-SNE Visualization"):
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    if len(embeddings_np) <= 30:
     raise ValueError("t-SNE requires more than 30 samples for perplexity=30.")

    tsne = TSNE(n_components=2, perplexity=30, max_iter=3000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels_np, cmap="tab20", s=15)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")
    # Function to denormalize images for visualization
def denormalize_image(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (image_tensor * std + mean).clamp(0, 1)
def find_top_k_neighbors_images(embeddings, query_index, k=5):
    similarity_scores = embeddings @ embeddings.T
    similarity_scores[query_index, query_index] = -1e9
    k = min(k, embeddings.size(0) - 1)
    top_k_indices = torch.topk(similarity_scores[query_index], k=k).indices
    return top_k_indices
def show_retrieved_images(query_idx,embeddings,images,labels,class_names, save_path="retrieved_images.png" ,k=5):
    actual_k = min(k, embeddings.size(0) - 1)
    topk_indices = find_top_k_neighbors_images(embeddings.clone(), query_idx, k=actual_k)
    fig, axes = plt.subplots(1, actual_k + 1, figsize=(3 * (actual_k + 1), 3))
    query_img_denorm = denormalize_image(images[query_idx]).permute(1, 2, 0).numpy()
    query_label = labels[query_idx].item()
    axes[0].imshow(query_img_denorm)
    axes[0].set_title(f"Query\n{class_names[query_label]}")
    axes[0].axis("off")
    for j, idx in enumerate(topk_indices):
       retrieved_img_denorm = denormalize_image(images[idx]).permute(1, 2, 0).numpy()
       retrieved_label = labels[idx].item()
       is_correct = (retrieved_label == query_label)
       axes[j+1].imshow(retrieved_img_denorm)
       axes[j+1].set_title(f"Retrieved\n{class_names[retrieved_label]}" , color="green" if is_correct else "red")
       axes[j+1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path,dpi=200)
    plt.close()
    print(f"Retrieved images saved to {save_path}")
def save_multiple_retrievals(embeddings, images, labels, class_names, num_queries=10,output_dir="retrieved_images", k=5,seed=42):
  os.makedirs(output_dir, exist_ok=True)
  total_images = len(images)
  num_queries = min(num_queries, total_images)
  random.seed(seed)
  query_indices = random.sample(range(total_images), num_queries)
  for query_idx in query_indices:
    save_path = os.path.join(output_dir, f"retrieval_{query_idx}.png")
    show_retrieved_images(query_idx, embeddings, images, labels, class_names, save_path=save_path, k=k)
def save_embeddings(embeddings, labels, save_dir="embeddings", prefix="data"):
    os.makedirs(save_dir, exist_ok=True)

    emb_path = os.path.join(save_dir, f"{prefix}_embeddings.npy")
    label_path = os.path.join(save_dir, f"{prefix}_labels.npy")

    np.save(emb_path, embeddings.cpu().numpy())
    np.save(label_path, labels.cpu().numpy())

    print(f"Embeddings saved to {emb_path}")
    print(f"Labels saved to {label_path}")

# Load embeddings and labels from .npy files
def load_embeddings(load_dir="embeddings", prefix="data"):
    emb_path = os.path.join(load_dir, f"{prefix}_embeddings.npy")
    label_path = os.path.join(load_dir, f"{prefix}_labels.npy")

    embeddings = torch.tensor(np.load(emb_path))
    labels = torch.tensor(np.load(label_path))

    print(f"Embeddings loaded from {emb_path}")
    print(f"Labels loaded from {label_path}")
    return embeddings, labels