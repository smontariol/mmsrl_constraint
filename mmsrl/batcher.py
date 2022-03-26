from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple
import collections

import torch

import mmsrl.utils


class Batcher:
    """
    Batch a group of sample together.

    Two new features are derived from the "text": its length and a mask.
    """

    def __init__(self, config: mmsrl.utils.dotdict, text_pad: int, label_pad: int, image_pad: int) -> None:
        """ Initialize a Batcher, using the provided value to pad text. """
        self.config: mmsrl.utils.dotdict = config
        self.text_pad: int = text_pad
        self.label_pad: int = label_pad
        self.image_pad: int = image_pad

    def process_text(self, batch: Dict[str, Any], target: str, prefix: str, pad: int) -> None:
        """ Build mask and text batch by padding sentences. """
        if target not in batch:
            return
        batch_size: int = len(batch[target])
        if isinstance(batch[target][0], torch.Tensor):
            length: torch.Tensor = torch.tensor([len(text) for text in batch[target]], dtype=torch.int64)
            max_seq_len: int = max(length.max().item(), self.config.get("min_text_len", 0))

            text: torch.Tensor = torch.empty((batch_size, max_seq_len), dtype=torch.int64)
            mask: torch.Tensor = torch.empty((batch_size, max_seq_len), dtype=torch.bool)
            for b, sentence in enumerate(batch[target]):
                text[b, :sentence.shape[0]] = sentence
                text[b, sentence.shape[0]:] = pad
                mask[b, :sentence.shape[0]] = True
                mask[b, sentence.shape[0]:] = False

            batch[target] = text
            batch[f"{prefix}length"] = length
            batch[f"{prefix}attention_mask"] = mask
        else:
            assert(isinstance(batch[target][0], list) and isinstance(batch[target][0][0], torch.Tensor))
            generate_mask2: bool = "decoder_answer_mask" in batch and target == "decoder_target"
            generate_mask3: bool = "decoder_constraint" in batch and target == "decoder_target"

            sup_length: int = len(batch[target][0])
            length: torch.Tensor = torch.tensor([[len(text) for text in sup] for sup in batch[target]], dtype=torch.int64)
            max_seq_len: int = max(length.max().item(), self.config.get("min_text_len", 0))

            text: torch.Tensor = torch.empty((batch_size, sup_length, max_seq_len), dtype=torch.int64)
            mask: torch.Tensor = torch.empty((batch_size, sup_length, max_seq_len), dtype=torch.bool)
            if generate_mask2:
                mask2: torch.Tensor = torch.empty((batch_size, sup_length, max_seq_len), dtype=torch.bool)
            if generate_mask3:
                vocabulary_size: int = batch["decoder_constraint"][0][0].shape[1]
                mask3: torch.Tensor = torch.empty((batch_size, sup_length, max_seq_len, vocabulary_size), dtype=torch.bool)
            for b, sup in enumerate(batch[target]):
                for i, sentence in enumerate(sup):
                    text[b, i, :sentence.shape[0]] = sentence
                    text[b, i, sentence.shape[0]:] = pad
                    mask[b, i, :sentence.shape[0]] = True
                    mask[b, i, sentence.shape[0]:] = False
                    if generate_mask2:
                        mask2[b, i, :sentence.shape[0]] = batch["decoder_answer_mask"][b][i]
                        mask2[b, i, sentence.shape[0]:] = False
                    if generate_mask3:
                        mask3[b, i, :sentence.shape[0]] = batch["decoder_constraint"][b][i]
                        mask3[b, i, sentence.shape[0]:] = False

            batch[target] = text
            batch[f"{prefix}length"] = length
            batch[f"{prefix}attention_mask"] = mask
            if generate_mask2:
                batch["decoder_answer_mask"] = mask2
            if generate_mask3:
                batch["decoder_constraint"] = mask3

    def process_entities(self, batch: Dict[str, Any]) -> None:
        """ Build mask and entities text batch by padding list of entities and their input ids. """
        batch_size: int = len(batch["entities_name"])
        max_num_entities: int = max(len(entities) for entities in batch["entities_name"])
        max_num_entities = max(max_num_entities, self.config.get("min_nb_ent", 0))

        if "labels" in batch:
            if batch["labels"][0].ndim == 2:
                assert(batch["labels"][0].shape[1] == len(mmsrl.utils.LABELS))
                labels: torch.Tensor = torch.empty((batch_size, max_num_entities, len(mmsrl.utils.LABELS)), dtype=torch.bool)
            else:
                labels: torch.Tensor = torch.empty((batch_size, max_num_entities), dtype=torch.int64)

            for b, entities in enumerate(batch["entities_name"]):
                labels[b, :len(entities)] = batch["labels"][b]
                labels[b, len(entities):] = self.label_pad
            batch["labels"] = labels

        mask: torch.Tensor = torch.empty((batch_size, max_num_entities), dtype=torch.bool)
        for b, entities in enumerate(batch["entities_name"]):
            mask[b, :len(entities)] = True
            mask[b, len(entities):] = False
        batch["entities_mask"] = mask

        if "entities_input_ids" in batch:
            lengths: torch.Tensor = torch.zeros((batch_size, max_num_entities), dtype=torch.int64)
            for b, entities in enumerate(batch["entities_input_ids"]):
                for i, entity in enumerate(entities):
                    lengths[b, i] = len(entity)
            max_seq_len: int = max(lengths.max().item(), self.config.get("min_ent_len", 0))

            text: torch.Tensor = torch.empty((batch_size, max_num_entities, max_seq_len), dtype=torch.int64)
            text_mask: torch.Tensor = torch.empty((batch_size, max_num_entities, max_seq_len), dtype=torch.bool)
            for b, entities in enumerate(batch["entities_input_ids"]):
                text[b, len(entities):] = self.text_pad
                text_mask[b, len(entities):] = False
                for i, entity in enumerate(entities):
                    text[b, i, :len(entity)] = entity
                    text[b, i, len(entity):] = self.text_pad
                    text_mask[b, i, :len(entity)] = True
                    text_mask[b, i, len(entity):] = False

            batch["entities_input_ids"] = text
            batch["entities_length"] = lengths
            batch["entities_attention_mask"] = text_mask

    def process_image(self, batch: Dict[str, Any]) -> None:
        """ Build width, height and mask by padding images. """
        batch_size: int = len(batch["image"])
        heights: torch.Tensor = torch.tensor([image.shape[1] for image in batch["image"]], dtype=torch.int64)
        widths: torch.Tensor = torch.tensor([image.shape[2] for image in batch["image"]], dtype=torch.int64)
        height: int = max(heights.max().item(), self.config.get("height_min", 0))
        width: int = max(widths.max().item(), self.config.get("width_min", 0))
        images: torch.Tensor = torch.full((batch_size, 3, height, width), self.image_pad, dtype=torch.float32)
        mask: torch.Tensor = torch.zeros((batch_size, height, width), dtype=torch.bool)

        for b, image in enumerate(batch["image"]):
            mask[b, :image.shape[1], :image.shape[2]] = True
            images[b, :, :image.shape[1], :image.shape[2]] = image

        batch["image"] = images
        batch["image_mask"] = mask
        batch["height"] = heights
        batch["width"] = widths

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ Batch the provided samples """
        batch = collections.defaultdict(list)
        for sample in samples:
            for key, value in sample.items():
                batch[key].append(value)

        self.process_text(batch, "input_ids", "", self.text_pad)
        self.process_text(batch, "vbert_input_ids", "vbert_", self.text_pad)
        self.process_text(batch, "caption_input_ids", "caption_", self.text_pad)
        self.process_text(batch, "decoder_input_ids", "decoder_", self.text_pad)
        self.process_text(batch, "decoder_target", "decoder_target_", self.label_pad)
        self.process_entities(batch)
        if batch.get("image"):
            self.process_image(batch)
        for feature in list(batch.keys()):
            if isinstance(batch[feature], list) and isinstance(batch[feature][0], torch.Tensor):
                batch[feature] = torch.stack(batch[feature])

        return batch
