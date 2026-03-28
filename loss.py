import torch
import torch.nn as nn
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    #forward method that computes the contrastive loss given two embeddings and a target label indicating whether they belong to the same class (1) or different classes (0)
    def forward(self, embedding1, embedding2, target):
    # calculate the Euclidean distance between the two embeddings
        euclidean_distance = torch.norm(embedding1 - embedding2, p=2, dim=1)
        loss=target * euclidean_distance.pow(2)+(1-target) * torch.clamp(self.margin - euclidean_distance, min=0.0).pow(2)
        return loss.mean()
    #Define a triplet loss class that computes the triplet loss given anchor, positive, and negative embeddings, along with a margin parameter that controls the separation between positive and negative pairs. The forward method calculates the distances between the anchor-positive and anchor-negative pairs, and computes the loss based on these distances.
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
    def forward(self, anchor, positive, negative):
    #compute the Euclidean distance between the anchor and positive embeddings, and between the anchor and negative embeddings
        pos_distance = torch.norm(anchor - positive, p=2, dim=1)
        neg_distance = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0.0)
        return loss.mean()
#Define a batch hard triplet loss function that computes the triplet loss for a batch of embeddings and their corresponding labels. 
def batch_hard_triplet_loss(embeddings, labels, margin=0.2):
    batch_size = embeddings.size(0)
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)
    labels=labels.view(-1, 1)
    pos_mask = (labels == labels.t())
    neg_mask = (labels != labels.t())

    eye_mask = torch.eye(batch_size, device=embeddings.device).bool()
    pos_mask = pos_mask & (~eye_mask)
    pos_distance=distance_matrix.clone()
    pos_distance[~pos_mask]=-1e9
    hardest_pos_distance = pos_distance.max(dim=1)[0]
    neg_distance=distance_matrix.clone()
    neg_distance[~neg_mask]=1e9
    hardest_neg_distance = neg_distance.min(dim=1)[0]
    valid_triplet = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if valid_triplet.sum() == 0:
      return embeddings.sum() * 0.0
    # compute the triplet loss using the hardest positive and hardest negative distances, and return the mean loss over valid triplets
    loss = torch.clamp(hardest_pos_distance - hardest_neg_distance + margin, min=0.0)
    return loss[valid_triplet].mean()
    