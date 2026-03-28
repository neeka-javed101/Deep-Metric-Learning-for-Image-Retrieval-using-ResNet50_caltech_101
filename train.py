import os
import copy
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import csv
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model import FeatureExtractor
from dataset import split_dataset, CustomDataset, ContrastiveDataset, TripletDataset, train_transform, test_transform
from reterival import get_embeddings, compute_recall_at_k, plot_tsne, save_multiple_retrievals
from embedding import save_embeddings
from loss import ContrastiveLoss, TripletLoss, batch_hard_triplet_loss
# Set random seeds for reproducibility

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_directory(base_dir, model_name):
    model_dir = os.path.join(base_dir, model_name)
    weight_dir = os.path.join(model_dir, "weights")
    embedding_dir = os.path.join(model_dir, "embeddings")
    graphs_dir = os.path.join(model_dir, "graphs")
    retrieval_dir = os.path.join(model_dir, "retrievals")
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(retrieval_dir, exist_ok=True)
    return weight_dir, embedding_dir, graphs_dir, retrieval_dir


def save_checkpoint(path, model, optimizer, epoch, best_recall_at_1):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_recall_at_1": best_recall_at_1,
    }, path)


def save_experiment_result(output_dir, experiment, loss_function, sampling_strategy,
                           epochs, batch_size, learning_rate, recall1, recall5):
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "experiment_results.csv")
    write_header = not os.path.exists(results_path)

    with open(results_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow([
                "Experiment","Loss Function", "Sampling Strategy","Epochs","Batch Size","Learning Rate","Recall@1","Recall@5",
            ])

        writer.writerow([experiment,loss_function,sampling_strategy, epochs, batch_size, learning_rate, recall1, recall5])
def evaluate(model, dataloader, device):
    embeddings, labels, images = get_embeddings(model, dataloader, device)
    recall1, recall5 = compute_recall_at_k(embeddings, labels)
    return recall1, recall5, embeddings, labels, images

# Training functions for different loss types
def train_contrastive_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    print("Inside train_contrastive_one_epoch")

    for batch_idx, (image1, image2, target) in enumerate(dataloader):
        if batch_idx == 0:
            print("First batch loaded")

        image1 = image1.to(device)
        image2 = image2.to(device)
        target = target.to(device).float()

        optimizer.zero_grad()

        embedding1 = model(image1)
        embedding2 = model(image2)

        loss = criterion(embedding1, embedding2, target)

        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def train_triplet_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for anchor, positive, negative in dataloader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        optimizer.zero_grad()
        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)
        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
# Batch-hard triplet training
def train_hard_triplet_one_epoch(model, dataloader, optimizer, device, margin=0.2):
    model.train()
    total_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        embeddings = model(images)
        loss = batch_hard_triplet_loss(embeddings, labels, margin)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Plotting functions
def plot_training_curves(train_losses, val_r1, val_r5, graphs_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'training_loss_curve.png'))
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(epochs, val_r1, label='Recall@1', marker='o')
    plt.plot(epochs, val_r5, label='Recall@5', marker='o')
    plt.title('Validation Recall@K Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Recall@K')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'validation_recall_curves.png'))
    plt.close()

# Main experiment function
def run_single_experiment(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weight_dir, embedding_dir, graphs_dir, retrieval_dir = make_directory("outputs", args.model_name)

    dataset = ImageFolder(args.data_dir)
    class_names = dataset.classes

    train_samples, val_samples, test_samples = split_dataset(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed
    )

    print(f"Number of training samples: {len(train_samples)}")
    print(f"Number of validation samples: {len(val_samples)}")
    print(f"Number of testing samples: {len(test_samples)}")
# Create DataLoaders for validation and testing
    val_dataset = CustomDataset(val_samples, transform=test_transform)
    test_dataset = CustomDataset(test_samples, transform=test_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
#    print("Sample training data:")
    if args.loss_type == "contrastive":
        train_dataset = ContrastiveDataset(train_samples, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        criterion = ContrastiveLoss(margin=args.margin)
        loss_function_name = "Contrastive Loss"
        sampling_strategy = "Random Pairs"

    elif args.loss_type == "triplet":
        train_dataset = TripletDataset(train_samples, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        criterion = TripletLoss(margin=args.margin)
        loss_function_name = "Triplet Loss"
        sampling_strategy = "Random Triplets"

    else:
        train_dataset = CustomDataset(train_samples, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        criterion = None
        loss_function_name = "Triplet Loss"
        sampling_strategy = "Hard Negative Mining"

    print("Creating model...")
    model = FeatureExtractor(embedding_dim=args.embedding_dimension).to(device)
    print("Model created successfully")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )
# 
    best_recall_at_1 = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    train_losses = []
    val_recall_at_1 = []
    val_recall_at_5 = []

    print("Starting training loop...")
    print("loss_type =", args.loss_type)
    print("num_epochs =", args.num_epochs)
    print("len(train_dataset) =", len(train_dataset))
    print("len(train_loader) =", len(train_loader))

# Training loop
    for epoch in range(1, args.num_epochs + 1):
        print(f"Entering epoch {epoch}")

        if args.loss_type == "contrastive":
            train_loss = train_contrastive_one_epoch(model, train_loader, criterion, optimizer, device)
        elif args.loss_type == "triplet":
            train_loss = train_triplet_one_epoch(model, train_loader, criterion, optimizer, device)
        else:
            train_loss = train_hard_triplet_one_epoch(
                model, train_loader, optimizer=optimizer, device=device, margin=args.margin
            )

        recall1, recall5, _, _, _ = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_recall_at_1.append(recall1)
        val_recall_at_5.append(recall5)

        print(f"Epoch {epoch}/{args.num_epochs} - Train Loss: {train_loss:.4f} - Val Recall@1: {recall1:.4f}, Recall@5: {recall5:.4f}")
        save_checkpoint(os.path.join(weight_dir, f"epoch_{epoch}.pth"), model, optimizer, epoch, best_recall_at_1)

        if recall1 > best_recall_at_1:
            best_recall_at_1 = recall1
            best_model_wts = copy.deepcopy(model.state_dict())
            save_checkpoint(os.path.join(weight_dir, "best_model.pth"), model, optimizer, epoch, best_recall_at_1)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    model.load_state_dict(best_model_wts)

    print("\nEvaluating best model on test set...")
    test_r1, test_r5, test_embeddings, test_labels, test_images = evaluate(model, test_loader, device)

    print(f"Test Recall@1 : {test_r1:.4f}")
    print(f"Test Recall@5 : {test_r5:.4f}")
# Save test embeddings and labels for t-SNE visualization and retrieval evaluation
    save_embeddings(
        os.path.join(embedding_dir, f"{args.loss_type}_test_embeddings.pt"),test_embeddings,test_labels,)
    plot_tsne(test_embeddings,test_labels,save_path=os.path.join(graphs_dir, f"{args.loss_type}_tsne.png"),title=f"{args.loss_type} t-SNE",
    )
    save_multiple_retrievals(
        test_embeddings, test_images, test_labels, class_names,
        num_queries=10, output_dir=retrieval_dir, k=5, seed=args.seed
    )
    plot_training_curves(train_losses, val_recall_at_1, val_recall_at_5, graphs_dir)

    exp_name = args.experiment_name if args.experiment_name is not None else args.model_name
    save_experiment_result(
        output_dir="outputs",experiment=exp_name,loss_function=loss_function_name, sampling_strategy=sampling_strategy, epochs=args.num_epochs,batch_size=args.batch_size,learning_rate=args.learning_rate,recall1=float(test_r1),
        recall5=float(test_r5),
    )
    print("\nAll results saved in:", os.path.join("outputs", args.model_name))
    # main function to parse arguments and run experiments
def main():
    parser = argparse.ArgumentParser(description="Train a feature extractor model using contrastive or triplet loss")
    parser.add_argument("--data_dir", type=str, required=True, help=r"C:\Users\Neeka Javeed\Desktop\Assignment#3\caltech-101")
    parser.add_argument("--model_name", type=str, default="resnet50_contrastive", help="Name of the model for saving checkpoints and embeddings",
           choices=["resnet50_contrastive", "resnet50_triplet", "resnet50_batch_hard_triplet"])
    parser.add_argument("--embedding_dimension", type=int, default=128, help="Dimension of the embedding space")
    parser.add_argument("--loss_type", type=str, choices=["contrastive", "triplet", "batch_hard_triplet"],
                        default="contrastive", help="Type of loss function to use for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin parameter for contrastive and triplet losses")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment label: Exp-1 / Exp-2 / Exp-3")
    parser.add_argument("--run_all", action="store_true", help="Run all three experiments automatically")
    args = parser.parse_args()
    if args.run_all:
        configs = [
            {
                "experiment_name": "Exp-1","model_name": "resnet50_contrastive","loss_type": "contrastive",
            },
            {
                "experiment_name": "Exp-2","model_name": "resnet50_triplet","loss_type": "triplet",
            },
            {
                "experiment_name": "Exp-3","model_name": "resnet50_batch_hard_triplet","loss_type": "batch_hard_triplet",
            },
        ]
        results_path = os.path.join("outputs", "experiment_results.csv")
        if os.path.exists(results_path):
            os.remove(results_path)
        for config in configs:
            args.experiment_name = config["experiment_name"]
            args.model_name = config["model_name"]
            args.loss_type = config["loss_type"]
            print(f"\nRunning {args.experiment_name} ...")
            run_single_experiment(args)
    else:
        run_single_experiment(args)
if __name__ == "__main__":
    main()
