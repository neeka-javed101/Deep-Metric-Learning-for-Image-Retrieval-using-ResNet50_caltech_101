import random
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
# split the dataset into training, validation, and testing sets
def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)

    class_to_samples = defaultdict(list)

    for path, label in dataset.samples:
        class_to_samples[label].append((path, label))

    train_samples, val_samples, test_samples = [], [], []

    for label, samples in class_to_samples.items():
        random.shuffle(samples)

        n_total = len(samples)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        train_samples.extend(samples[:n_train])
        val_samples.extend(samples[n_train:n_train + n_val])
        test_samples.extend(samples[n_train + n_val:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    return train_samples, val_samples, test_samples
train_transform= transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
class CustomDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
           
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
# Dataset for contrastive training
class ContrastiveDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.labels = [label for _, label in self.samples]

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        self.unique_labels = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.samples)

    def load_image(self, idx):
        path, _ = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        _, label = self.samples[idx]
        image1 = self.load_image(idx)

        # 50% same class, 50% different class
        if random.random() < 0.5 and len(self.label_to_indices[label]) > 1:
            pair_idx = idx
            while pair_idx == idx:
                pair_idx = random.choice(self.label_to_indices[label])
            target = 1.0
        else:
            negative_label = random.choice(self.unique_labels)
            while negative_label == label:
                negative_label = random.choice(self.unique_labels)
            pair_idx = random.choice(self.label_to_indices[negative_label])
            target = 0.0

        image2 = self.load_image(pair_idx)

        return image1, image2, torch.tensor(target, dtype=torch.float32)

# Dataset for triplet training
class TripletDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.labels = [label for _, label in self.samples]

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        self.unique_labels = list(self.label_to_indices.keys())

        # only keep anchors from classes that have at least 2 images
        self.valid_anchor_indices = [
            idx for idx, (_, label) in enumerate(self.samples)
            if len(self.label_to_indices[label]) > 1
        ]

        if len(self.valid_anchor_indices) == 0:
            raise ValueError("TripletDataset requires at least one class with 2+ images.")
        if len(self.unique_labels) < 2:
            raise ValueError("TripletDataset requires at least 2 different classes.")

    def __len__(self):
        return len(self.valid_anchor_indices)

    def load_image(self, idx):
        path, _ = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        anchor_idx = self.valid_anchor_indices[idx]
        _, anchor_label = self.samples[anchor_idx]

        positive_candidates = [i for i in self.label_to_indices[anchor_label] if i != anchor_idx]
        positive_idx = random.choice(positive_candidates)

        negative_label = random.choice(self.unique_labels)
        while negative_label == anchor_label:
            negative_label = random.choice(self.unique_labels)

        negative_idx = random.choice(self.label_to_indices[negative_label])

        anchor = self.load_image(anchor_idx)
        positive = self.load_image(positive_idx)
        negative = self.load_image(negative_idx)

        return anchor, positive, negative
class BatchHardDataset(Dataset):
     def __init__(self, samples, transform=None, classes_per_batch=8, samples_per_class=4):
        """
        Batch-hard sampling:
        - Select N classes
        - For each class pick K samples
        - Total batch size = N * K
        """
        self.samples = samples
        self.transform = transform
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class

        self.labels = [label for _, label in self.samples]

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        self.unique_labels = list(self.label_to_indices.keys())

        # keep only labels with enough samples
        self.valid_labels = [
            label for label in self.unique_labels
            if len(self.label_to_indices[label]) >= self.samples_per_class
        ]

        if len(self.valid_labels) < self.classes_per_batch:
            raise ValueError("Not enough classes with sufficient samples for batch hard training.")

     def __len__(self):
        # arbitrary length (depends on training iterations)
        return len(self.samples)

     def load_image(self, idx):
        path, _ = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

     def __getitem__(self, idx):
        # randomly sample classes
        selected_labels = random.sample(self.valid_labels, self.classes_per_batch)

        batch_images = []
        batch_labels = []

        for label in selected_labels:
            indices = random.sample(self.label_to_indices[label], self.samples_per_class)

            for i in indices:
                img = self.load_image(i)
                batch_images.append(img)
                batch_labels.append(label)

        batch_images = torch.stack(batch_images)
        batch_labels = torch.tensor(batch_labels)

        return batch_images, batch_labels