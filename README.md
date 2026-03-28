# Deep Metric Learning for Image Retrieval

This project implements and compares three deep metric learning approaches for image retrieval using the Caltech-101 dataset.

## Methods

- Contrastive Loss (Random Pairs)
- Triplet Loss (Random Triplets)
- Batch-Hard Triplet Loss (Hard Negative Mining)

## Features

- ResNet-50 feature extractor
- Embedding dimension: 128
- Recall@K evaluation
- t-SNE visualization
- Retrieval visualization
- Training and validation curves

## Results

| Method               | Recall@1 | Recall@5 |
|----------------------|----------|----------|
| Contrastive Loss     | 0.70     | 0.80     |
| Triplet Loss         | 0.80     | 0.80     |
| Batch-Hard Triplet   | 0.80     | 0.80     |

## Observations

- Contrastive loss gives baseline performance
- Triplet loss improves embedding learning
- Batch-hard triplet provides best results due to hard sample mining

## Dataset

- Caltech-101
- Image size: 224x224
- Split: 70% train, 15% validation, 15% test

## How to Run

1. Install dependencies:
2. Train models:
3. Evaluate retrieval:
## Conclusion

Batch-hard triplet learning produces the most discriminative embeddings and achieves the best retrieval performance.

## Author

Neeka Javed
