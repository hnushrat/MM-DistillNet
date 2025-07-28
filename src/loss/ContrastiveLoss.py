from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive Loss"""
    def __init__(self, temperature = 0.4):
        super(ContrastiveLoss, self).__init__()
        self.T = temperature

    # TO-DO

    def forward(self, features_t, features_s, indices):
        '''
        indices -> across batch the indices of top-k vectors in input (rgb, thermal, depth);
        dimensions(batch, 112, 36)
        '''
        B, C, T = features_t.shape
        K = indices.shape[1]

        # Gather embeddings for the given indices
        # (B, C, K)
        gathered1 = torch.gather(features_t, 2, indices.unsqueeze(1).expand(-1, C, -1))
        gathered2 = torch.gather(features_s, 2, indices.unsqueeze(1).expand(-1, C, -1))

        # Flatten batch and index dim for easier loss computation
        # New shape: (B*K, C)
        z1 = gathered1.transpose(1, 2).reshape(-1, C)  # shape: (B*K, C)
        z2 = gathered2.transpose(1, 2).reshape(-1, C)  # shape: (B*K, C)

        # Normalize for cosine similarity
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute logits: (B*K, B*K)
        logits = torch.matmul(z1, z2.T) / self.T

        # Labels: positive pairs are on the diagonal
        labels = torch.arange(z1.shape[0], device=logits.device)

        # Contrastive InfoNCE loss
        loss = F.cross_entropy(logits, labels)

        return loss