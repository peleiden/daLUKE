#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
import os

import torch
from pelutils import log, Levels, Parser, EnvVars, get_timestamp, set_seeds

from daluke.serialize import load_from_archive, save_to_archive, COLLECT_OUT, TRAIN_OUT
from daluke.ner import load_dataset, load_model
from daluke.ner.model import mutate_for_ner
from daluke.ner.training import TrainNER
from daluke.ner.data import Split
from daluke.ner.evaluation import type_distribution

DATASET_ARGUMENTS = {
    "dataset":    {"default": "DaNE", "help": "Which dataset to use. Currently, only DaNE supported"},
    "wikiann-path":
        {"default": "wikiann",
            "help": "Path to folder containing WikiANN-da data set. Only needed if --dataset WikiANN.\n"\
                    "Dataset was downloaded from https://github.com/afshinrahimi/mmner"
        },
    "plank-path":
        {"default": "plank",
            "help": "Path to the folder containing B. Plank data set. Only needed if --dataset Plank.\n"\
                    "Dataset was downloaded from https://github.com/bplank/danish_ner_transfer"
        },
}

ARGUMENTS = {
    "model": {
        "help": ".tar.gz file containing the model, metadata, and entity vocab as generated by collect_modelfile",
        "default": os.path.join("local_data", COLLECT_OUT)
    },
    "lr":              {"default": 1e-5, "type": float},
    "epochs":          {"default": 5, "type": int},
    "batch-size":      {"default": 16, "type": int},
    "max-entity-span": {"help": "Max. length of spans used in data. If not given, use the one in pre-training metadata",
                           "default": None, "type": int},
    "max-entities":    {"help": "Max. enitites in each example. If not given, use the one in pre-training metadata",
                           "default": None, "type": int},
    "warmup-prop":     {"default": 0.06, "type": float},
    "weight-decay":    {"default": 0.01, "type": float},
    "dropout":         {"default": None, "type": float},
    "words-only":      {"action": "store_true", "help": "Use only start and end token CWR's for classification"},
    "entities-only":   {"action": "store_true", "help": "Use only CER for classification"},
    "dataset":         {"help": "Which dataset to use. Currently, only DaNE supported", "default": "DaNE"},
    "eval":            {"help": "Run evaluation on dev. set after each epoch", "action": "store_true"},
    "quieter":         {"help": "Don't show debug logging", "action": "store_true"},
    "loss-weight":     {"help": "Weight loss contributions by class frequency", "action": "store_true"},
    **DATASET_ARGUMENTS,
}

def run_experiment(args: dict[str, Any]):
    log.section("Beginnig", args["name"])
    set_seeds(seed=0)
    assert not (args["words_only"] and args["entities_only"]), "--words-only and --entities-only cannot be used together"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    entity_vocab, metadata, state_dict = load_from_archive(args["model"])
    state_dict, ent_embed_size = mutate_for_ner(state_dict, mask_id=entity_vocab["[MASK]"]["id"])

    log(f"Loading dataset {args['dataset']} ...")
    dataset = load_dataset(args, metadata, device)
    dataloader = dataset.build(Split.TRAIN, args["batch_size"])
    dev_dataloader = dataset.build(Split.DEV, args["batch_size"]) if args["eval"] else None

    log("Loading model ...")
    model = load_model(
        state_dict,
        dataset,
        metadata,
        device,
        words_only = args["words_only"],
        entities_only = args["entities_only"],
        entity_embedding_size = ent_embed_size,
        dropout = args["dropout"],
    )

    log(f"Starting training of DaLUKE for NER on {args['dataset']}")
    training = TrainNER(
        model,
        dataloader,
        dataset,
        device         = device,
        epochs         = args["epochs"],
        lr             = args["lr"],
        warmup_prop    = args["warmup_prop"],
        weight_decay   = args["weight_decay"],
        dev_dataloader = dev_dataloader,
        loss_weight    = args["loss_weight"],
    )
    # Log important information out
    log.debug(training.model)
    log.debug(training.scheduler)
    log.debug(training.optimizer)
    dataset.document(dataloader, Split.TRAIN)
    type_distribution(dataset.data[Split.TRAIN].annotations)

    results = training.run()

    if args["eval"]:
        log("True dev. set distributions")
        results.dev_true_type_distribution = type_distribution(dataset.data[Split.DEV].annotations)
        log("True dev. set distributions")
        results.train_true_type_distribution = type_distribution(dataset.data[Split.TRAIN].annotations)
    os.makedirs(args["location"], exist_ok=True)
    results.save(args["location"])
    outpath = os.path.join(args["location"], TRAIN_OUT)
    save_to_archive(outpath, entity_vocab, metadata, model)
    log("Training complete, saved model archive to", outpath)

if __name__ == '__main__':
    with log.log_errors, EnvVars(TOKENIZERS_PARALLELISM=str(not "Tue").lower()):
        parser = Parser(ARGUMENTS, name="daluke-ner-"+get_timestamp(for_file=True), multiple_jobs=True)
        experiments = parser.parse()
        log.configure(
            os.path.join(parser.location, "daluke-train-ner.log"), "Finetune daLUKE for Danish NER",
            print_level=Levels.INFO if experiments[0]["quieter"] else Levels.DEBUG,
        )
        parser.document_settings()
        for exp in experiments:
            run_experiment(exp)
