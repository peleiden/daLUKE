from __future__ import annotations
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Iterator, Any
from itertools import chain

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from transformers import AutoTokenizer
from danlp.datasets import DDT

from daluke.data import Entities, Example, BatchedExamples, Words, get_special_ids

@dataclass
class NEREntities(Entities):
    start_pos: torch.Tensor
    end_pos: torch.Tensor
    labels: torch.Tensor
    true_spans: list[tuple[int, int]]

    @classmethod
    def build_from_entities(cls, ent: Entities, labels,true_spans: list[tuple[int, int]]):
        breakpoint()
        start_pos = NotImplementedError
        end_pos = NotImplementedError
        return cls(ent.ids, ent.attention_mask, ent.N, ent.spans, ent.pos, start_pos, end_pos, labels, true_spans)

@dataclass
class NERExample(Example):
    """
    A single data example for Named Entity Recognition
    """
    entities: NEREntities
    text_num: int

@dataclass
class NERBatchedExamples(BatchedExamples):
    @classmethod
    def build(
        cls,
        examples: list[NERExample],
        device:   torch.device,
        cut_extra_padding: bool=True,
    ):
        words, entities = cls.collate(examples, device=device, cut=cut_extra_padding)
        return cls(words, entities, word_mask_labels, word_mask, ent_mask_labels, ent_mask)

class Split(IntEnum):
    TRAIN = 0
    DEV = 1
    TEST = 2

class NERDataset(ABC):
    # feature_names = ("ent_start_pos", "ent_end_pos", "word_ids", "ent_ids", "word_seg_ids", "ent_seg_ids",
        # "ent_pos_ids", "word_att_mask", "ent_att_mask", "labels", "idx")

    null_label: str = None
    labels: tuple[str] = None

    def __init__(self,
            entity_vocab: dict,
            base_model: str,
            max_seq_length: int,
            max_entities: int,
            max_entity_span: int,
            device: torch.device,
        ):
        self.device = device
        self.entity_vocab = entity_vocab
        self.max_seq_length = max_seq_length
        self.max_entities = max_entities
        self.max_entity_span = max_entity_span

        self.label_to_idx = {label: i for i, label in enumerate(self.all_labels)}

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.sep_id, self.cls_id, self.pad_id = get_special_ids(self.tokenizer)

        # To be set in build method
        self.examples: list[Example] = None
        self.texts: list[list[str]] = None
        self.annotations: list[list[str]] = None

    @abstractmethod
    def build(self, split: Split, batch_size: int) -> DataLoader:
        pass

    @property
    def all_labels(self):
        L = list() if self.null_label is None else [self.null_label]
        return L + list(self.labels)

    def _build_examples(self, sent_bounds: list[list[int]]) -> list[dict[str, Any]]:
        examples = list()
        for i, (text, annotation, bounds) in enumerate(zip(self.texts, self.annotations, sent_bounds)):
            text_token_ids: list[list[int]] = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            # We might have to split some sentences to respect the maximum sentence length
            bounds = self._add_extra_sentence_boundaries(bounds, text_token_ids, annotation)
            # TODO: Consider the current sentence splitting: Do we throw away context in situations where we actually have sentence-document information
            for j, end in enumerate(bounds):
                start = bounds[j-1] if j else 0
                true_entity_full_word_spans = self._segment_entities(annotation[start: end])
                # The cumulative length of each word in units of subwords
                cumlength = np.cumsum([len(t) for t in text_token_ids[start: end]])
                # Save the spans of entities as they are in the token list
                true_entity_spans = {(cumlength[entstart]-1, cumlength[entend-1]): ann
                    for (entstart, entend), ann in true_entity_full_word_spans.items()}

                assert all(e-s <= self.max_entity_span for e, s in true_entity_spans),\
                        f"Example {i}, sentence {j} contains an entity true_longer than limit of {self.max_entity_span} tokens. Text:\n{text}"
                assert len(true_entity_spans) < self.max_entities,\
                        f"Example {i}, sentence {j} contains {len(true_entity_spans)} entities, but only {self.max_entities} are allowed. Text:\n{text}"

                all_entity_spans = self._generate_all_entity_spans(true_entity_spans, text_token_ids[start: end], cumlength)
                # We dont use the entity id: We just mark that it *is* an entity
                entity_ids = [self.entity_vocab["[UNK]"]["id"] for _ in range(len(all_entity_spans))]
                entity_labels = [self.label_to_idx[true_entity_spans.get(span, self.null_label)] for span in all_entity_spans]

                entities = Entities.build(
                    torch.IntTensor(entity_ids),
                    all_entity_spans,
                    max_entities    = self.max_entities,
                    max_entity_span = self.max_entity_span
                )
                words = Words.build(
                    # Flatten structure of [[subword], [subword], ... ]
                    torch.IntTensor(list(chain(*text_token_ids[start: end]))),
                    max_len = self.max_seq_length,
                    sep_id  = self.sep_id,
                    cls_id  = self.cls_id,
                    pad_id  = self.pad_id,
                )
                examples.append(
                    NERExample(
                        words    = words,
                        entities = NEREntities.build_from_entities(
                            entities,
                            true_spans = list(true_entity_spans.keys()),
                            labels = entity_labels,
                        ),
                        text_num = i,
                    )
                )
        return examples

    def collate(self, batch: list[tuple[int, NERExample]]) -> NERBatchedExamples:
        return NERBatchedExamples.build(
            [ex for _, ex in batch],
            self.device,
            cut_extra_padding = True,
        )

    def _generate_all_entity_spans(self, true_spans: dict[tuple[int, int], str], token_ids: list[list[int]], cumlength: np.ndarray) -> list[tuple[int, int]]:
        possible_spans = list()
        # Spans are (1, 2), (1, 3), (1, 4) ... (1, N), (2, 3), (2, 4), ... (N-1, N)
        for i in range(len(token_ids)):
            for j in range(i+1, len(token_ids)):
                if (i, j) not in true_spans and (cumlength[j] - cumlength[i]) <= self.max_entity_span:
                    possible_spans.append((i, j))
        # TODO: Consider whether we should select this another way than randomly (first 2-spans, then 3-spans, ...?)
        random.shuffle(possible_spans)
        # Make sure we return <= max_entities while including the true spans
        return [*possible_spans[:self.max_entities-len(true_spans)], *true_spans.keys()]

    def _segment_entities(self, annotation: list[str]) -> dict[tuple[int, int], str]:
        """
        Reads entity annotation in IOB format and finds entity spans mapped to entity type
        """
        spans = dict()
        start = None
        ent_type = None
        for i, ann in enumerate(annotation):
            # Ignore "O"
            if ann == self.null_label:
                assert start is None, "Found O label, but another entity was not ended"
                continue
            tag, typ_ = ann.split("-")
            # Might be end if either (1) last in sentence or (2) followed by explicit start or (3) followed by another annotation type
            if i+1 == len(annotation) or annotation[i+1] == self.null_label or\
                    (next_ann := annotation[i+1].split("-"))[0] == "B" or next_ann[1] != typ_:
                assert ent_type is None or typ_ == ent_type, "Entity seemed to change annotation during span - this should not be possible"
                spans[(i if start is None else start, i+1)] = typ_
                # We ended entity, look for a new one
                start = None
                ent_type = None
            # Might be a beginning if it is either (1) explicitly marked (2) starts sentence (3) follows another annotation type
            elif tag == "B" or not i or annotation[i-1] == self.null_label or\
                    annotation[i-1].split("-")[1] != typ_:
                assert start is None, "Found start entity while another one was not ended"
                start = i
                ent_type = typ_
        return spans

    def _add_extra_sentence_boundaries(self, bounds: list[int], text_token_ids: list[list[int]], annotation: list[str]) -> list[int]:
        # Check whether we should add another sentence bound by splitting one of the sentences
        might_need_split = True
        while might_need_split:
            for i, bound in enumerate(bounds):
                # Sum up subword lengths for this sentence
                sent_start = bounds[i-1] if i else 0
                sentence_cumlength = np.cumsum([len(tokens) for tokens in text_token_ids[sent_start:bound]])
                if sentence_cumlength[-1] + 2 > self.max_seq_length: # +2 for start and end tokens
                    # Use bool cast to int to find the number of words that give a sum under the limit.
                    split_candidate = (sentence_cumlength + 2 < self.max_seq_length).sum()
                    # TODO: Maybe split more intelligently such as checking whether this split candidate breaks up an entity
                    bounds.insert(i, sent_start + split_candidate)
                    break
            else:
                might_need_split = False
        return bounds
        # feature_objects: list[InputFeatures] = convert_examples_to_features(
        #     list(zip(self.texts, self.annotations, sent_bounds)),
        #     self.all_labels,
        #     self.tokenizer,
        #     max_seq_length=self.max_sentence_len,
        #     max_entity_length=self.max_entities,
        #     max_mention_length=self.max_entity_span,
        # )
        # # Convert to dict using only the relevant fields, as the data is highly flexible

        # fields = ("entity_start_positions", "entity_end_positions", "word_ids", "entity_ids", "word_segment_ids",
        #     "entity_segment_ids", "entity_position_ids", "word_attention_mask", "entity_attention_mask", "labels", "example_index")
        # self.entity_spans = [f_obj.original_entity_spans for f_obj in feature_objects]
        # return [
        #     {fname: getattr(f_obj, field) for fname, field in zip(self.feature_names, fields)}
        #         for f_obj in feature_objects
        # ]

    # def _collate(self, batch: Iterator[tuple[int, dict[str, Any]]]):
    #     """
    #     Collect dataset examples into tensors and pad them for sequence classification
    #     """
    #     paddings = {"word_ids": self.tokenizer.pad_token_id, "ent_pos_ids": -1, "labels": -1}
    #     collated = dict()
    #     for feature in self.feature_names:
    #         if feature == "idx": continue
    #         tensors = [torch.tensor(x[1][feature], dtype=torch.long) for x in batch]
    #         pad_val = paddings.get(feature, 0)
    #         collated[feature] = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_val)
    #     if self.current_split != Split.TRAIN:
    #         collated.pop("labels")
    #         collated["idx"] = [x[1]["idx"] for x in batch]
    #         collated["spans"] = [self.entity_spans[x[0]] for x in batch]
    #     return collated

class DaNE(NERDataset):
    null_label = "O"
    labels = ("LOC", "PER", "ORG", "MISC")

    def build(self, split: Split, batch_size: int) -> DataLoader:
        self.texts, self.annotations = DDT().load_as_simple_ner(predefined_splits=True)[split.value]
        # Sadly, we do not have access to where the DaNE sentences are divided into articles, so we let each sentence be an entire text.
        sentence_boundaries = [[len(s)] for s in self.texts]
        self.examples = self._build_examples(sentence_boundaries)
        self.examples = self.examples[:10]
        return DataLoader(list(enumerate(self.features)), batch_size=batch_size, collate_fn=self.collate, shuffle=split==split.TRAIN)
