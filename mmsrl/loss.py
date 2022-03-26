from typing import Optional

import torch


class Loss(torch.nn.Module):
    def __init__(self, config, label_weights):
        super().__init__()
        self.config = config
        self.label_weights = label_weights
        # reduction none because we weight it with class weights and then we do the average ourself.
        if self.config.multilabel:
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.loss = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=self.config.get("label_smoothing", 0))

    def forward(self, logits: torch.Tensor, labels: Optional[torch.Tensor] = None):
        if labels is None:
            loss = None
        elif self.config.multilabel:
            loss = self.loss(logits, labels)
            loss = loss*self.label_weights[labels%4] # %4: artisanal way to avoid the -100 when indexing
            loss = loss.mean()
        else:
            loss = self.loss(logits.view(labels.numel(), 4), labels.view(-1)) # numel = nb of elements in labels in total (product of dimensions)
            loss = loss*self.label_weights[labels.view(-1)%4] # %4: artisanal way to avoid the -100 when indexing
            loss = loss.mean()

        if self.config.multilabel:
            prediction = torch.nn.functional.sigmoid(logits, dim=2)
        else:
            prediction = torch.nn.functional.softmax(logits, dim=2)

        return loss, prediction
