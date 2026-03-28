import os
import torch
import matplotlib.pyplot  as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import FeatureExtractor



from dataset import (
    split_dataset,
    ContrastiveDataset,
    TripletDataset,
    CustomDataset,
    BatchHardDataset,
    train_transform,
)
# ---------------------------
# CONFIG
# ---------------------------
DATA_PATH = r"C:\Users\Neeka Javeed\Desktop\Assignment#3\caltech-101"
NUM_SAMPLES = 20

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTRASTIVE_OUTPUT_DIR = os.path.join(BASE_DIR, "contrastive_results")
TRIPLET_OUTPUT_DIR = os.path.join(BASE_DIR, "triplet_results")

os.makedirs(CONTRASTIVE_OUTPUT_DIR, exist_ok=True)
os.makedirs(TRIPLET_OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Denormalization
# ---------------------------
def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)

# ---------------------------
# Load dataset
# ---------------------------
dataset = ImageFolder(DATA_PATH)
train_samples, _, _ = split_dataset(dataset)

contrastive_dataset = ContrastiveDataset(train_samples, transform=train_transform)
triplet_dataset = TripletDataset(train_samples, transform=train_transform)

print("Total contrastive samples:", len(contrastive_dataset))
print("Total triplet samples:", len(triplet_dataset))


# CONTRASTIVE
print("\n========== CONTRASTIVE DATASET ==========")

contrastive_labels_file = os.path.join(CONTRASTIVE_OUTPUT_DIR, "labels.txt")

with open(contrastive_labels_file, "w") as f:
    f.write("Index, Label, Type\n")

    for i in range(NUM_SAMPLES):
        _, _, label = contrastive_dataset[i]
        pair_type = "Same Class" if label.item() == 1 else "Different Class"
        print(f"Contrastive Sample {i}: Label = {label.item()} ({pair_type})")
        f.write(f"{i}, {label.item()}, {pair_type}\n")

print("\n--- Saving Contrastive Image Pairs ---")

for i in range(NUM_SAMPLES):
    img1, img2, label = contrastive_dataset[i]

    img1 = denormalize(img1)
    img2 = denormalize(img2)

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(img1.permute(1, 2, 0))
    plt.title("Image 1")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2.permute(1, 2, 0))
    plt.title(f"Image 2\nLabel = {label.item()}")
    plt.axis("off")

    plt.suptitle("Same Class" if label.item() == 1 else "Different Class")
    plt.tight_layout()

    save_path = os.path.join(CONTRASTIVE_OUTPUT_DIR, f"pair_{i}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


# TRIPLET

print("\n========== TRIPLET DATASET ==========")

triplet_info_file = os.path.join(TRIPLET_OUTPUT_DIR, "triplet_info.txt")

with open(triplet_info_file, "w") as f:
    f.write("Index, Description\n")

    for i in range(NUM_SAMPLES):
        _anchor, _positive, _negative = triplet_dataset[i]
        print(f"Triplet Sample {i}: Anchor / Positive / Negative")
        f.write(f"{i}, Anchor-Positive: Same Class | Anchor-Negative: Different Class\n")

print("\n--- Saving Triplet Images ---")

for i in range(NUM_SAMPLES):
    anchor, positive, negative = triplet_dataset[i]

    anchor = denormalize(anchor)
    positive = denormalize(positive)
    negative = denormalize(negative)

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(anchor.permute(1, 2, 0))
    plt.title("Anchor")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(positive.permute(1, 2, 0))
    plt.title("Positive\n(Same Class)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(negative.permute(1, 2, 0))
    plt.title("Negative\n(Different Class)")
    plt.axis("off")

    plt.suptitle(f"Triplet {i}")
    plt.tight_layout()

    save_path = os.path.join(TRIPLET_OUTPUT_DIR, f"triplet_{i}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
# ADVANCED BATCH-HARD VISUALIZATION

BATCH_HARD_VIS_OUTPUT_DIR = os.path.join(BASE_DIR, "batch_hard_visualization")
os.makedirs(BATCH_HARD_VIS_OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
EMBEDDING_DIM = 128
NUM_WORKERS = 0
MODEL_NAME = "resnet50_batch_hard_triplet"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataLoader for batch-hard visualization
batch_hard_dataset = BatchHardDataset(train_samples, transform=train_transform)

batch_hard_loader = DataLoader(
    batch_hard_dataset,
    batch_size=1,
    shuffle=True
)
# Load trained batch-hard model
checkpoint_path = os.path.join(BASE_DIR, "outputs", MODEL_NAME, "weights", "best_model.pth")

if not os.path.exists(checkpoint_path):
    print("Batch-hard model not found at:", checkpoint_path)
else:
    model = FeatureExtractor(embedding_dim=EMBEDDING_DIM).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded batch-hard model from:", checkpoint_path)

    mages, labels = next(iter(batch_hard_loader))

# remove outer dimension added by DataLoader(batch_size=1)
images, labels = next(iter(batch_hard_loader))
images = images.squeeze(0).to(device)
labels = labels.squeeze(0).to(device)

with torch.no_grad():
    embeddings = model(images)

    distance_matrix = torch.cdist(embeddings, embeddings, p=2)

    labels_col = labels.view(-1, 1)
    pos_mask = (labels_col == labels_col.t())
    neg_mask = (labels_col != labels_col.t())

    eye_mask = torch.eye(labels.size(0), device=device).bool()
    pos_mask = pos_mask & (~eye_mask)

    saved_count = 0

    for i in range(labels.size(0)):
        if not pos_mask[i].any() or not neg_mask[i].any():
            continue

        # Hardest positive: farthest same-class sample
        pos_distances = distance_matrix[i].clone()
        pos_distances[~pos_mask[i]] = -1e9
        hardest_pos_idx = torch.argmax(pos_distances).item()
        hardest_pos_distance = distance_matrix[i, hardest_pos_idx].item()

        # Hardest negative: closest different-class sample
        neg_distances = distance_matrix[i].clone()
        neg_distances[~neg_mask[i]] = 1e9
        hardest_neg_idx = torch.argmin(neg_distances).item()
        hardest_neg_distance = distance_matrix[i, hardest_neg_idx].item()

        anchor_img = denormalize(images[i].cpu())
        pos_img = denormalize(images[hardest_pos_idx].cpu())
        neg_img = denormalize(images[hardest_neg_idx].cpu())

        anchor_label = labels[i].item()
        pos_label = labels[hardest_pos_idx].item()
        neg_label = labels[hardest_neg_idx].item()

        print(f"\nBatch-Hard Visual Sample {saved_count}")
        print(f"Anchor label: {anchor_label}")
        print(f"Hardest positive label: {pos_label}, distance: {hardest_pos_distance:.4f}")
        print(f"Hardest negative label: {neg_label}, distance: {hardest_neg_distance:.4f}")

        plt.figure(figsize=(9, 3))

        plt.subplot(1, 3, 1)
        plt.imshow(anchor_img.permute(1, 2, 0))
        plt.title(f"Anchor\nLabel={anchor_label}")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(pos_img.permute(1, 2, 0))
        plt.title(f"Hardest Positive\nLabel={pos_label}\nDist={hardest_pos_distance:.3f}")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(neg_img.permute(1, 2, 0))
        plt.title(f"Hardest Negative\nLabel={neg_label}\nDist={hardest_neg_distance:.3f}")
        plt.axis("off")

        plt.tight_layout()

        save_path = os.path.join(BATCH_HARD_VIS_OUTPUT_DIR, f"batch_hard_triplet_{saved_count}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()

        saved_count += 1
        if saved_count >= NUM_SAMPLES:
            break

print("Batch-Hard visualization saved in:", BATCH_HARD_VIS_OUTPUT_DIR)
print("Contrastive results saved in:", CONTRASTIVE_OUTPUT_DIR)
print("Triplet results saved in:", TRIPLET_OUTPUT_DIR)
