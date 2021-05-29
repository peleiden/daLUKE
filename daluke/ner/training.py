from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup

from pelutils import log, DataStorage

from .evaluation import evaluate_ner, type_distribution, NER_Results
from .data import Split, NERDataset

@dataclass
class TrainResults(DataStorage):
    d_losses: list[float]
    c_losses: list[float]
    losses: list[float]
    running_evaluations: list[NER_Results]
    pred_distributions: list[dict[str, int]]
    true_type_distribution: dict[str, int]


    subfolder = "train-results"

class TrainNER:
    # These layers should not be subject to weight decay
    no_decay = {"bias", "LayerNorm.weight"}

    def __init__(self,
            model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
            dataset: NERDataset,
            device: torch.device,
            epochs: int,
            lr: float = 1e-5,
            warmup_prop: float = 0.06,
            weight_decay: float = 0.01,
            dev_dataloader: torch.utils.data.DataLoader | None = None,
            loss_weight: bool = False,
        ):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.dataset = dataset
        self.dev_dataloader = dev_dataloader
        self.epochs = epochs
        # Create optimizer
        params = list(model.named_parameters())
        optimizer_parameters = [
             {"params": self._get_optimizer_params(params, do_decay=True), "weight_decay": weight_decay},
             {"params": self._get_optimizer_params(params, do_decay=False), "weight_decay": 0.0}
        ]
        self.optimizer = AdamW(
            optimizer_parameters,
            lr           = lr,
            betas        = (0.9, 0.98),
            correct_bias = False,
        )
        # Create LR scheduler
        num_updates = epochs * len(self.dataloader)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(warmup_prop * num_updates), num_updates)
        if loss_weight:
            counts = torch.zeros(len(dataset.all_labels))
            for _, e in self.dataloader.dataset:
                # Do count on the non-padded labels
                for label, count in zip(*e.entities.labels[:e.entities.N].unique(return_counts=True)):
                    counts[label] += count

        self.discriminator_criterion = nn.BCELoss()
        self.classifier_criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=1/counts[1:].to(device) if loss_weight else None)

    def run(self):
        self.model.train()
        res = TrainResults(
            d_losses = list(),
            c_losses = list(),
            losses   = list(),
            running_evaluations    = list(),
            pred_distributions     = list(),
            true_type_distribution = dict(),
        )
        for i in range(self.epochs):
            for j, batch in enumerate(self.dataloader):
                discriminator_out, class_scores, is_ent = self.model(batch)
                d_loss = self.discriminator_criterion(discriminator_out, (batch.entities.labels != 0).to(torch.float32).unsqueeze(2))
                c_loss = self.classifier_criterion(class_scores, batch.entities.labels.view(-1)[is_ent]-1)
                # Set nan loss to 0. Happens when no entities in batch
                if torch.isnan(c_loss):
                    loss = d_loss
                else:
                    loss = d_loss + c_loss
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

                res.d_losses.append(d_loss.item())
                res.c_losses.append(c_loss.item())
                res.losses.append(loss.item())
                log.debug(f"Epoch {i} / {self.epochs-1}, batch: {j} / {len(self.dataloader)-1}. LR: {self.scheduler.get_last_lr()[0]:.2e}. DLoss: {d_loss.item()}. CLoss: {c_loss.item():.5f}. Loss: {loss.item():.5f}.")
            # Perform running evaluation
            if self.dev_dataloader is not None:
                log("Evaluating on development set ...")
                dev_results = evaluate_ner(self.model, self.dev_dataloader, self.dataset, self.device, Split.DEV, also_no_misc=False)
                res.running_evaluations.append(dev_results)
                res.pred_distributions.append(type_distribution(dev_results.preds))
                self.model.train()
        return res


    def _get_optimizer_params(self, params: list, do_decay: bool) -> list:
        # Only include the parameter if do_decay has reverse truth value of the parameter being in no_decay
        save = lambda n: not do_decay == any(nd in n for nd in self.no_decay)
        return [p for n, p in params if p.requires_grad and save(n)]
