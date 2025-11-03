import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import torchvision.models as models
from torchvision.models.resnet import Bottleneck
import torchvision.models as models
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor

import numpy as np
import random
import cv2
import math

from .triplet_loss import (
    _get_anchor_positive_triplet_mask,
    _get_anchor_negative_triplet_mask,
    _get_triplet_mask,
)

# URLs for loading ImageNet-pretrained weights if requested
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


class ResNet(models.ResNet):
    """
    ResNet backbone with an MLP head for embeddings and a comparator head.

    Parameters
    ----------
    block : nn.Module
        Residual block type (e.g., `Bottleneck` for ResNet-50/101/152).
    layers : list[int]
        Stage depths (e.g., [3,4,6,3] for ResNet-50).
    output_dim : int
        Final embedding dimension (feature size returned by `forward`).

    Attributes added on top of torchvision ResNet
    --------------------------------------------
    avgpool : nn.AvgPool2d
        Spatial pooling tuned for tall person crops.
    fc / bn_fc / relu_fc / fc_out :
        Two-layer MLP producing the final embedding of dimension `output_dim`.
    fc_compare : nn.Linear
        A small head mapping |e0 - e1| to a 1-d logit (for pairwise scoring).
    """

    def __init__(self, block, layers, output_dim):
        super(ResNet, self).__init__(block, layers)

        self.name = "ResNet"

        # Replace the default pooling with a kernel suited to (256x128) crops.
        self.avgpool = nn.AvgPool2d((8, 4), stride=1)

        # Two-layer projection head: 2048 -> 1024 -> output_dim (for ResNet-50)
        # (First input dim is 512 * expansion; expansion=4 for Bottleneck.)
        self.fc = nn.Linear(512 * block.expansion, 1024)
        self.bn_fc = nn.BatchNorm1d(1024)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(1024, output_dim)

        # Initialize BN1d in the added head to identity (weight=1, bias=0)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Comparator head for pair similarity (logit output)
        self.fc_compare = nn.Linear(output_dim, 1)

    def forward(self, x):
        """
        Standard forward pass producing an `output_dim` embedding per input.

        Input
        -----
        x : Tensor of shape (N, 3, H, W)

        Steps
        -----
        - Base ResNet stem and stages conv1..layer4
        - AvgPool with kernel (8, 4)
        - Flatten and pass through fc -> bn -> relu -> fc_out
        """
        x = self.conv1(x)   # stem
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # ResNet stages
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)           # global-ish pooling adapted for aspect ratio
        x = x.view(x.size(0), -1)     # flatten to (N, C)
        x = self.fc(x)                # 2048 -> 1024
        x = self.bn_fc(x)             # normalize
        x = self.relu_fc(x)           # nonlinearity
        x = self.fc_out(x)            # -> output_dim embedding

        return x

    def test_rois(self, image, rois):
        """Tests the ROIs on a particular image. Should be inside image.

        Notes
        -----
        - `image` is expected as a 4D tensor shaped like a batch (N,C,H,W);
          this function will take channel-first crops using `build_crops`.
        - Each ROI `r` is (x0, y0, x1, y1) in pixel coordinates.
        - Returns embeddings for the stacked crops.
        """
        x = self.build_crops(image, rois)   # crop + resize + stack to a batch
        x = Variable(x)
        # TODO: fp16 currently under development phase
        return self.forward(x)

    def compare(self, e0, e1, train=False):
        """
        Compare two embeddings by absolute difference and map to a logit/score.

        Parameters
        ----------
        e0, e1 : Tensor
            Embedding tensors of identical shape (N, D).
        train : bool
            If True, return raw logits (for BCEWithLogits); otherwise apply
            sigmoid to return probabilities in [0,1].

        Returns
        -------
        Tensor
            (N, 1) logits (train=True) or probabilities (train=False).
        """
        out = torch.abs(e0 - e1)   # elementwise absolute difference
        out = self.fc_compare(out) # linear mapping to a 1-d logit
        
        if not train:
            out = torch.sigmoid(out)    # convert to probability for inference
        return out

    def build_crops(self, image, rois):
        """
        Crop and preprocess ROIs from `image` for embedding extraction.

        Parameters
        ----------
        image : Tensor
            A 4D tensor (N,C,H,W); only the first batch item (index 0) is used.
        rois : Iterable
            Each ROI `r` is a 4-tuple (x0, y0, x1, y1) in pixels.

        Returns
        -------
        Tensor
            A 4D tensor (R, 3, 256, 128) stacked from resized crops and moved to CUDA.
        """
        res = []
        # Compose: (CHW float tensor in [0,1] or raw) -> PIL -> resize -> tensor
        trans = Compose([ToPILImage(), Resize((256, 128)), ToTensor()])
        for r in rois:
            x0 = int(r[0])
            y0 = int(r[1])
            x1 = int(r[2])
            y1 = int(r[3])

            # Ensure x-range/y-range are non-degenerate (at least 1 pixel wide/tall)
            if x0 == x1:
                if x0 != 0:
                    x0 -= 1
                else:
                    x1 += 1
            if y0 == y1:
                if y0 != 0:
                    y0 -= 1
                else:
                    y1 += 1
            
            # Take crop from the first image in the batch (index 0)
            im = image[0, :, y0:y1, x0:x1]
            im = trans(im)  # -> (3, 256, 128), float in [0,1]
            res.append(im)

        res = torch.stack(res, 0)  # (R, 3, 256, 128)
        res = res.cuda()           # move to GPU (assumes CUDA context)
        return res

    def sum_losses(self, batch, loss, margin, prec_at_k):
        """For Pretraining

        Function for preatrainind this CNN with the triplet loss. Takes a sample of N=PK images, P different
        persons, K images of each. K=4 is a normal parameter.

        [!] Batch all and batch hard should work fine. Take care with weighted triplet or cross entropy!!

        Args:
            batch (list): [images, labels], images are Tensor of size (N,H,W,C), H=224, W=112, labels Tensor of
            size (N)
        """
        # Unpack and CUDA-move inputs/labels. The code expects batch[0][0] and batch[1][0].
        inp = batch[0][0]
        inp = Variable(inp).cuda()

        labels = batch[1][0]
        labels = labels.cuda()

        # Forward pass to get embeddings (N, D)
        embeddings = self.forward(inp)

        if loss == "cross_entropy":
            # Build all valid triplets (i, j, k) from labels using a mask
            m = _get_triplet_mask(labels).nonzero()
            e0 = []
            e1 = []
            e2 = []
            for p in m:
                e0.append(embeddings[p[0]])
                e1.append(embeddings[p[1]])
                e2.append(embeddings[p[2]])
            e0 = torch.stack(e0, 0)
            e1 = torch.stack(e1, 0)
            e2 = torch.stack(e2, 0)

            # Binary classification: positive pair vs. negative pair
            out_pos = self.compare(e0, e1, train=True)
            out_neg = self.compare(e0, e2, train=True)

            # Targets for BCE-with-logits
            tar_pos = Variable(torch.ones(out_pos.size(0)).view(-1, 1).cuda())
            tar_neg = Variable(torch.zeros(out_pos.size(0)).view(-1, 1).cuda())

            loss_pos = F.binary_cross_entropy_with_logits(out_pos, tar_pos)
            loss_neg = F.binary_cross_entropy_with_logits(out_neg, tar_neg)

            total_loss = (loss_pos + loss_neg) / 2

        elif loss == "batch_all":
            # Batch-all triplet: use every valid (anchor, positive, negative)
            m = _get_triplet_mask(labels).nonzero()
            e0 = []
            e1 = []
            e2 = []
            for p in m:
                e0.append(embeddings[p[0]])
                e1.append(embeddings[p[1]])
                e2.append(embeddings[p[2]])
            e0 = torch.stack(e0, 0)
            e1 = torch.stack(e1, 0)
            e2 = torch.stack(e2, 0)
            total_loss = F.triplet_margin_loss(e0, e1, e2, margin=margin, p=2)
        elif loss == "batch_hard":
            # Batch-hard triplet: select hardest positive/negative per anchor.
            # Compute pairwise squared distances between embeddings.
            n = embeddings.size(0)
            m = embeddings.size(0)
            d = embeddings.size(1)

            x = embeddings.data.unsqueeze(1).expand(n, m, d)
            y = embeddings.data.unsqueeze(0).expand(n, m, d)

            dist = torch.pow(x - y, 2).sum(2)   # (N, N) squared L2 distances

            # Masks for valid positives/negatives
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

            # Distances for positives (invalid entries are zeroed)
            pos_dist = dist * mask_anchor_positive

            # For negatives, add a large constant to invalid entries to avoid picking them
            max_val = torch.max(dist)
            neg_dist = dist + max_val * (1.0 - mask_anchor_negative)

            # For each anchor i, select:
            #   pos = argmax_j pos_dist[i, j] (hardest positive)
            #   neg = argmin_k neg_dist[i, k] (hardest negative)
            triplets = []
            for i in range(dist.size(0)):
                pos = torch.max(pos_dist[i], 0)[1].item()
                neg = torch.min(neg_dist[i], 0)[1].item()
                triplets.append((i, pos, neg))

            # Gather the corresponding embeddings and compute triplet loss
            e0 = []
            e1 = []
            e2 = []
            for p in triplets:
                e0.append(embeddings[p[0]])
                e1.append(embeddings[p[1]])
                e2.append(embeddings[p[2]])
            e0 = torch.stack(e0, 0)
            e1 = torch.stack(e1, 0)
            e2 = torch.stack(e2, 0)
            total_loss = F.triplet_margin_loss(e0, e1, e2, margin=margin, p=2)

        elif loss == "weighted_triplet":
            # Weighted-triplet scheme:
            # 1) Compute pairwise (L2) distances for the batch
            dist = []
            # Construct distance matrix column-by-column (less memory-frugal but clear)
            for e in embeddings:
                ee = torch.cat([e.view(1, -1) for _ in range(embeddings.size(0))], 0)
                dist.append(F.pairwise_distance(embeddings, ee))
            dist = torch.cat(dist, 1)

            # Positive mask (same label) and distances
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
            pos_dist = dist * Variable(mask_anchor_positive.float())

            # Negative mask (different label) and distances
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
            neg_dist = dist * Variable(mask_anchor_negative.float())

            # Compute per-anchor soft weights:
            #   positives: softmax over pos distances
            #   negatives: softmin (i.e., softmax(-x)) over neg distances
            pos_weights = Variable(torch.zeros(dist.size()).cuda())
            neg_weights = Variable(torch.zeros(dist.size()).cuda())
            for i in range(dist.size(0)):
                mask = torch.zeros(dist.size()).byte().cuda()
                mask[i] = 1
                pos_weights[mask_anchor_positive & mask] = F.softmax(
                    pos_dist[mask_anchor_positive & mask], 0
                )
                neg_weights[mask_anchor_negative & mask] = F.softmin(
                    neg_dist[mask_anchor_negative & mask], 0
                )

            # Do not backpropagate through weights (treat them as constants)
            pos_weights = pos_weights.detach()
            neg_weights = neg_weights.detach()

            # Aggregate weighted distances for positives/negatives per anchor
            pos_weight_dist = pos_dist * pos_weights
            neg_weight_dist = neg_dist * neg_weights

            # Margin-based loss on the weighted sums
            triplet_loss = torch.clamp(
                margin
                + pos_weight_dist.sum(1, keepdim=True)
                - neg_weight_dist.sum(1, keepdim=True),
                min=0,
            )
            total_loss = triplet_loss.mean()
        else:
            # Unsupported configuration
            raise NotImplementedError("Loss: {}".format(loss))

        losses = {}

        if prec_at_k:
            # Compute pairwise squared distances again for retrieval metric
            n = embeddings.size(0)
            m = embeddings.size(0)
            d = embeddings.size(1)

            x = embeddings.data.unsqueeze(1).expand(n, m, d)
            y = embeddings.data.unsqueeze(0).expand(n, m, d)

            dist = torch.pow(x - y, 2).sum(2)
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
            _, indices = torch.sort(dist, dim=1)

            # Count hits among top-k nearest neighbors (excluding self)
            num_hit = 0
            num_ges = 0
            for i in range(dist.size(0)):
                d = mask_anchor_positive[i].nonzero().view(-1, 1)
                ind = indices[i][: prec_at_k + 1]   # include self at rank 0
                same = d == ind
                num_hit += same.sum()
                num_ges += prec_at_k
            
            # Store as a CUDA tensor wrapped in Variable for consistency
            k_loss = torch.Tensor(1)
            k_loss[0] = num_hit / num_ges
            losses["prec_at_k"] = Variable(k_loss.cuda())

        # Always return the primary training loss
        losses["total_loss"] = total_loss

        return losses

    def load_pretrained_dict(self, state_dict):
        """Load the pretrained weights and ignore the ones where size does not match

        Rationale
        ---------
        When the architecture head differs from the ImageNet classifier (which is
        typical in ReID), only load overlapping keys with identical tensor shapes.
        """
        # Filter the incoming state dict to keep only same-shaped matching keys
        pretrained_state_dict = {
            k: v
            for k, v in state_dict.items()
            for kk, vv in self.state_dict().items()
            if k == kk and v.size() == vv.size()
        }

        # Merge filtered weights into current state dict and load
        updated_state_dict = self.state_dict()
        updated_state_dict.update(pretrained_state_dict)
        self.load_state_dict(updated_state_dict)


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args
    ----
    pretrained : bool
        If True, initialize from ImageNet-pretrained weights.

    Notes
    -----
    - This function instantiates our `ResNet` wrapper with `Bottleneck` blocks
      and stage configuration [3,4,6,3], then (optionally) loads ImageNet weights.
    - Only weights whose shapes match the current model will be loaded.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_pretrained_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model
