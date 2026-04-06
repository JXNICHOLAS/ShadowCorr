"""
Segment-ID-level encoder (word-embedding style).

Each segment ID gets its own learned embedding vector. A voxel's embedding is
the mean of its segment IDs' embeddings. Training via contrastive loss on
segment-ID co-occurrence makes IDs that frequently co-occur in voxels closer.

Analogous to Word2Vec: word = segment ID, sentence = voxel's segment set,
co-occurrence count = how many voxels two segment IDs share.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentIDEncoder(nn.Module):
    """
    Each segment ID has a learned embedding (like word embeddings).
    A voxel's embedding = mean of its segment IDs' L2-normalized embeddings.
    Output lives on the unit sphere so the full embedding space is utilized.
    """

    def __init__(self, max_segments: int, embed_dim: int = 12, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_segments = max_segments
        self.segment_embedding = nn.Embedding(max_segments, embed_dim)
        nn.init.normal_(self.segment_embedding.weight, mean=0.0, std=0.5)

    def forward(self, segment_lists):
        if not segment_lists:
            return torch.zeros((0, self.embed_dim), device=self.segment_embedding.weight.device)

        dev = self.segment_embedding.weight.device
        voxel_embeddings = []
        for seg_list in segment_lists:
            if len(seg_list) == 0:
                voxel_embeddings.append(torch.zeros(self.embed_dim, device=dev))
            else:
                seg_ids = torch.tensor(seg_list, dtype=torch.long, device=dev)
                seg_embs = self.segment_embedding(seg_ids)  # (num_segs, embed_dim)
                voxel_emb = seg_embs.mean(dim=0)
                voxel_embeddings.append(voxel_emb)

        stacked = torch.stack(voxel_embeddings)
        return F.normalize(stacked, p=2, dim=1)  # project onto unit sphere


def create_segment_encoder(max_segments: int, embed_dim: int = 12, **kwargs):
    """Create a SegmentIDEncoder (word-embedding style)."""
    return SegmentIDEncoder(max_segments, embed_dim=embed_dim)
