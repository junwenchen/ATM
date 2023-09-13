import torch as torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
class Contrastive_Loss(torch.nn.Module):
    def __init__(self):
        super(Contrastive_Loss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, target):
        return self.ce_loss(x, target)


class LogSoftmax(torch.nn.Module):
    def __init__(self, dim):
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x, a):
        nll = -F.log_softmax(x, self.dim, _stacklevel=5)
        return (nll * a / a.sum(1, keepdim=True).clamp(min=1)).sum(dim=1).mean()


class NCELoss(torch.nn.Module):
    def __init__(self, batch_size=4096):
        super(NCELoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size = len(x)
        target = torch.arange(batch_size).cuda()
        x = torch.cat((x, x.t()), dim=1)
        return self.ce_loss(x, target)


# class HLoss(nn.Module):
#     """
#         returning the negative entropy of an input tensor
#     """
#     def __init__(self, is_maximization=False):
#         super(HLoss, self).__init__()
#         self.is_neg = is_maximization

#     def forward(self, x, qtype=None):
#         score = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
#         if self.is_neg:
#             # b = 1.0 * b.sum()          # summation over batches         
#             # loss = 1.0 * score.sum(dim=1).mean()     # summation over batches, mean over batches   
#             if qtype is not None:
#                 if qtype.sum() != 0:
#                     loss = score.sum(dim=1).sum()/qtype.sum()   
#                 else:
#                     loss = score.sum(dim=1).sum()
#             else:
#                 loss = score.sum(dim=1).mean()
#         else:
#             # b = -1.0 * b.sum()
#             loss = -1.0 * score.sum(dim=1).mean()     # summation over batches, mean over batches
#         return loss
class HLoss(nn.Module):
    """
        returning the negative entropy of an input tensor
    """
    def __init__(self, is_maximization=False):
        super(HLoss, self).__init__()
        self.is_neg = is_maximization

    def forward(self, x, qtype=None):
        score = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        if self.is_neg:
            # b = 1.0 * b.sum()          # summation over batches         
            # loss = 1.0 * score.sum(dim=1).mean()     # summation over batches, mean over batches   
            if qtype is not None:
                if qtype.sum() != 0:
                    loss = (score*qtype.unsqueeze(1)).sum(dim=1).sum()/qtype.sum()   
                else:
                    loss = (score*qtype.unsqueeze(1)).sum(dim=1).sum()
            else:
                loss = score.sum(dim=1).mean()
        else:
            # b = -1.0 * b.sum()
            loss = -1.0 * score.sum(dim=1).mean()     # summation over batches, mean over batches
        return loss