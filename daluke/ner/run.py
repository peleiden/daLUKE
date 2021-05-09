#!/usr/bin/env python3
from __future__ import annotations
import os

import torch
from transformers import AutoConfig
from pelutils import log, Levels, Parser

import daluke.ner.data as datasets

from daluke.collect_modelfile import OUT_FILE as collect_out
from daluke.serialize import load_from_archive, save_to_archive
from daluke.ner.model import NERDaLUKE, mutate_for_ner
from daluke.ner.training import TrainNER
from daluke.ner.data import NERDataset, Split

OUT_FILE = "daluke_ner.tar.gz"

ARGUMENTS = {
    "model": {
        "help": "directory or .tar.gz file containing the model, metadata and entity vocab generated by collect_modelfile",
        "default": os.path.join("local_data", collect_out)
    },
    "lr":              {"default": 1e-5, "type": float},
    "epochs":          {"default": 5, "type": int},
    "batch-size":      {"default": 16, "type": int},
    "warmup-prop":     {"default": 0.06, "type": float},
    "weight-decay":    {"default": 0.01, "type": float},

    "quieter": {"help": "Don't show debug logging", "action": "store_true"},
    "cpu":     {"help": "Run experiment on cpu",    "action": "store_true"},
    "dataset": {"help": "Which dataset to use. Currently, only DaNE supported", "default": "DaNE"},
}

def run_experiment(args: dict[str, str]):
    device = torch.device("cpu") if args["cpu"] or not torch.cuda.is_available() else torch.device("cuda")
    entity_vocab, metadata, state_dict = load_from_archive(args["model"])
    state_dict, ent_embed_size = mutate_for_ner(state_dict, mask_id=entity_vocab["[MASK]"]["id"])

    log.debug("Loading dataset ...")
    dataset = getattr(datasets, args["dataset"])
    dataset: NERDataset = dataset(
        entity_vocab,
        base_model      = metadata["base-model"],
        max_seq_length  = metadata["max-seq-length"],
        max_entities    = metadata["max-entities"],
        max_entity_span = metadata["max-entity-span"],
        device          = device,
    )
    dataloader = dataset.build(Split.TRAIN, args["batch_size"])

    log.debug("Loading model ...")
    bert_config = AutoConfig.from_pretrained(metadata["base-model"])
    model = NERDaLUKE(
        len(dataset.all_labels),
        bert_config,
        ent_vocab_size = 2, # Same reason as mutate_for_ner
        ent_embed_size = ent_embed_size,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    log(f"Starting training of DaLUKE for NER on {args['dataset']}")
    training = TrainNER(
        model,
        dataloader,
        device          = device,
        epochs          = args["epochs"],
        lr              = args["lr"],
        warmup_prop     = args["warmup_prop"],
        weight_decay    = args["weight_decay"],
    )
    log.debug(training.model)
    log.debug(training.scheduler)
    log.debug(training.optimizer)

    results = training.run()

    os.makedirs(args["location"], exist_ok=True)
    results.save(args["location"])
    outpath = os.path.join(args["location"], OUT_FILE)
    save_to_archive(outpath, entity_vocab, metadata, model)
    log("Training complete, saved model archive to", outpath)

if __name__ == '__main__':
    with log.log_errors:
        parser = Parser(ARGUMENTS, name="daLUKE-NER-finetune", multiple_jobs=True)
        experiments = parser.parse()
        parser.document_settings()
        log.configure(
            os.path.join(parser.location, "daluke_train_ner.log"), "Finetune daLUKE for Danish NER",
            print_level=Levels.INFO if experiments[0]["quieter"] else Levels.DEBUG
        )
        for exp in experiments:
            run_experiment(exp)
