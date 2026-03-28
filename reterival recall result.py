import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from dataset import split_dataset, CustomDataset, test_transform
from model import FeatureExtractor
from reterival import get_embeddings, compute_recall_at_k

# CONFIG

DATA_PATH = r"C:\Users\Neeka Javeed\Desktop\Assignment#3\caltech-101"
MODEL_NAME = "resnet50_triplet"  # change if needed

# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset (test split)

dataset = ImageFolder(DATA_PATH)

_, _, test_samples = split_dataset(dataset)
test_samples = test_samples[:200]

test_dataset = CustomDataset(test_samples, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Load trained model

model = FeatureExtractor(embedding_dim=128).to(device)

checkpoint_path = f"outputs/{MODEL_NAME}/weights/best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded model from:", checkpoint_path)


# Get embeddings
embeddings, labels, _ = get_embeddings(model, test_loader, device)

print("Embeddings shape:", embeddings.shape)
# Compute Recall
recall1, recall5 = compute_recall_at_k(embeddings, labels)

print("\n========== RETRIEVAL RESULTS ==========")
print(f"Recall@1: {recall1:.4f}")
print(f"Recall@5: {recall5:.4f}")
for i in range(10):
    scores = embeddings @ embeddings.t()
    scores[i][i] = -1e9

    top_k = torch.topk(scores[i], k=5).indices

    print(f"\nQuery {i} label: {labels[i].item()}")
    print("Top-10 labels:", labels[top_k].tolist())
    print("\n========== RETRIEVAL TABLE ==========")
print(f"{'Query':<10}{'Label':<10}{'Top-10 Labels':<35}{'R@1':<5}{'R@5':<5}")
print("-" * 70)

recall1_total = 0
recall5_total = 0
num_queries = 10  # change if needed

similarity_matrix = embeddings @ embeddings.t()

for i in range(num_queries):
    scores = similarity_matrix[i].clone()
    scores[i] = -1e9

    top_k_indices = torch.topk(scores, k=5).indices
    top_k_labels = labels[top_k_indices].tolist()
    query_label = labels[i].item()

    # Recall@1
    r1 = 1 if top_k_labels[0] == query_label else 0

    # Recall@5
    r5 = 1 if query_label in top_k_labels else 0

    recall1_total += r1
    recall5_total += r5

    print(f"{i:<10}{query_label:<10}{str(top_k_labels):<35}{r1:<5}{r5:<5}")