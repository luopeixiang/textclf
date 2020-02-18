import torch
import torch.nn.functional as F

from textclf.config.criterion import (
    CriterionConfig,
    CrossEntropyLossConfig,
    FocalLossConfig
)


class Criterion(object):
    def __init__(self, config: CriterionConfig):
        self.config = config

    def __call__(self, logits, targets):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            loss
        """
        raise NotImplementedError

    def _reduce(self, method, loss):
        """
        :param method (str): mean, sum or none
        :param loss: shape of [batch_size,]
        """
        assert method in ["mean", "sum", "none"]
        if method == "mean":
            loss = loss.mean()
        elif method == "sum":
            loss = loss.sum()
        return loss


class CrossEntropyLoss(Criterion):
    def __init__(self, config: CrossEntropyLossConfig):
        super(CrossEntropyLoss, self).__init__(config)
        self.use_cuda = config.use_cuda
        self.weight = torch.tensor(config.weight).float() if config.weight else None
        if self.use_cuda and self.weight is not None:
            self.weight = self.weight.cuda()
        self.reduction = config.reduction
        self.label_smooth_eps = config.label_smooth_eps

    def __call__(self, logits, targets):
        if self.label_smooth_eps is None:
            loss = F.cross_entropy(
                logits, targets,
                weight=self.weight, reduction=self.reduction
            )
        else:
            nll = -F.log_softmax(logits, dim=1)
            label_smooth_mat = torch.ones_like(nll)*(
                self.label_smooth_eps/(logits.size(1)-1)
            )
            label_smooth_mat.scatter_(1, targets.unsqueeze(1), 1-self.label_smooth_eps)
            if self.weight is not None:
                weight_mat = self.weight.unsqueeze(0).expand_as(nll)
                loss = label_smooth_mat * nll * weight_mat
            else:
                loss = label_smooth_mat * nll

            loss = self._reduce(self.reduction, loss.sum(dim=1))

        return loss


class FocalLoss(Criterion):
    """Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002"""

    def __init__(self, config: FocalLossConfig):
        super(FocalLoss, self).__init__(config)
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.reduction = config.reduction

    def __call__(self, logits, targets):
        idx = targets.view(-1, 1).long()
        mask = torch.zeros_like(logits)
        mask = mask.scatter_(1, idx, 1).byte()
        logits = torch.softmax(logits, dim=-1)
        loss = -self.alpha * torch.pow((1 - logits), self.gamma) * \
            (logits + 1.e-9).log()  # 防止溢出所以加上1.e-9
        loss = loss.masked_select(mask)
        loss = self._reduce(self.reduction, loss)
        return loss
