from typing import List

import json
import os
import pathlib
import pandas
import random
import collections

import torch
import PIL
import torchvision
import mmsrl.utils


class MemeDataset(torch.utils.data.IterableDataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(self, config, split, is_eval, tokenizer=None, ofa_task=None, image_processor=None, image_features_loader=None, vbert_tokenizer = None):
        """
        Initializes a Dataset class

        Args:
            dataname in  ['uspolitics','covid19', 'all']
            split in ['train', 'val', 'test', 'unseen_test']
            tokenizer (transformers.tokenizer): Transformers tokenizer
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.is_eval = is_eval
        self.ofa_task = ofa_task
        self.has_labels = ('unseen' not in split)

        if os.environ.get("DATA_PATH", "") == "":
            raise RuntimeError("Environment variable DATA_PATH is not set. Run the following command: export DATA_PATH=\"$PWD/../data\"")
        data_path = pathlib.Path(os.environ['DATA_PATH'])
        annot_file = data_path / "mmsrl" / self.config.dataname / 'annotations' / f"{split}.jsonl"
        with annot_file.open('r') as json_file:
            self.dataset = list(map(json.loads, json_file))

        self.images_path = data_path / "mmsrl" / self.config.dataname / 'images'
        if self.config.get('use_caption') and self.config.dataname == "all":
            caption_path = data_path / "mmsrl" / self.config.dataname / "captions.csv"
            self.captions_df = pandas.read_csv(caption_path, sep='\t', header=None)

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        self.image_transform = image_processor
        if image_features_loader is not None:
            self.image_features_loader = image_features_loader
        if vbert_tokenizer is not None:
            self.vbert_tokenizer = vbert_tokenizer
        elif self.config.model in ["ofa", "mmf"]:
            self.image_transform = torchvision.transforms.Compose([
                lambda image: image.convert("RGB"),
                self.resize_image,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ])

        if self.config.model == "ofa" and not self.config.get("group_image_entities"):
            self.dataset_len = 0
            for sample in self.dataset:
                if "entity_list" in sample:
                    self.dataset_len += len(set(sample["entity_list"]))
                else:
                    self.dataset_len += sum(len(sample[label]) for label in mmsrl.utils.LABELS)
                    # If we ever remove entities appearing twice in samples:
                    #entities = set()
                    #for label in mmsrl.utils.LABELS:
                    #    entities.update(sample[label])
                    #self.dataset_len += len(entities)

        if self.ofa_task:
            self.ofa_bos = torch.LongTensor([self.ofa_task.src_dict.bos()])
            self.ofa_eos = torch.LongTensor([self.ofa_task.src_dict.eos()])

        if self.config.get("maybe_is", "") in ["yes", "no"]:
            self.snli_answers = ["no", "maybe", "yes"]
        else:
            self.snli_answers = ["no", "yes"]

        if self.config.get("constrain_output"):
            assert(self.config.model == "ofa")
            assert(not self.config.get("only_score_answer"))
            assert(not self.config.get("only_train_answer"))
            assert(not self.config.get("ofa_classification_head"))
            assert(self.config.get("decoder_input") is not None and self.config.decoder_input == "{label}")
            if self.config.ofa_task == "vqa_gen":
                self.build_trie(mmsrl.utils.LABELS)
            else:
                self.build_trie(self.snli_answers)

        self.seed(0)
        self.compute_subsample(epoch=0)

    def seed(self, id: int):
        self.common_rng = random.Random(0)
        self.rng = random.Random(id)

    def __len__(self):
        """returns the length of dataframe"""
        if not self.is_eval and self.config.get("batch_per_epoch"):
            return self.config.batch_per_epoch * self.config.batch_size
        elif self.config.model == "ofa" and not self.config.get("group_image_entities"):
            return self.dataset_len
        else:
            # Approximation due to sub-sampling entities deleting empty samples.
            return len(self.dataset)

    def get_label_weights(self):
        if self.config.get("class_weights"):
            label_weights = torch.zeros(4, dtype=torch.float32, requires_grad=False)
            for sample in self.dataset:
                for i, label in enumerate(mmsrl.utils.LABELS):
                    label_weights[i] += len(sample[label])
            label_weights = 1/label_weights
            label_weights /= label_weights.mean() # normalise
        else:
            label_weights = torch.ones(4, dtype=torch.float32, requires_grad=False)
        return label_weights

    def compute_subsample(self, epoch):
        if self.is_eval:
            return
        if self.config.get("cyclic_subsample") is not None:
            config = mmsrl.utils.dotdict(self.config.cyclic_subsample[epoch % len(self.config.cyclic_subsample)])
        elif epoch>0 and not self.config.get("subsample_labels", "").startswith("interpolate"):
            return
        else:
            config = self.config

        frequencies = self.config.label_frequencies
        if config.get("subsample_labels", "micro") == "micro":
            target_frequencies = frequencies
        elif config.subsample_labels == "macro":
            target_frequencies = [0.25]*4
        elif config.subsample_labels == "interpolate_micro_to_macro":
            initial_frequencies = frequencies
            final_frequencies = [0.25]*4
            factor = epoch / (self.config.max_epoch - 1)
            target_frequencies = [
                    factor * final + (1 - factor) * initial
                    for initial, final in zip(initial_frequencies, final_frequencies)]
        elif config.subsample_labels == "affine":
            target_frequencies = [frequency * config.frequency_factor + config.frequency_bias for frequency in frequencies]
        elif config.subsample_labels == "target":
            target_frequencies = config.frequency_target
        else:
            raise RuntimeError(f"Unknown value for config.subsample_labels: {config.subsample_labels}")
        target_frequencies[3] *= config.get("subsample_other", 1)
        target_frequencies = [frequency / sum(target_frequencies) for frequency in target_frequencies]
        sampling_frequencies = [frequency / target for frequency, target in zip(frequencies, target_frequencies)]
        sampling_frequencies = [frequency / sum(sampling_frequencies) for frequency in sampling_frequencies]
        self.subsample_probs = [min(sampling_frequencies) / frequency for frequency in sampling_frequencies]
        
        repr_frequencies = ", ".join(f"{f:5.3f}" for f in target_frequencies)
        repr_subsample = ", ".join(f"{p:5.3f}" for p in self.subsample_probs)
        if self.split == "train":
            print(f"\033[33mSubsampling factors are [{repr_subsample}]\033[0m (target frequencies: [{repr_frequencies}])")

    def resize_image(self, image):
        if max(image.size) > self.config.image_max_edge:
            factor: float = self.config.image_max_edge / max(image.size)
            height: int = round(image.size[0] * factor)
            width: int = round(image.size[1] * factor)
            image = torchvision.transforms.functional.resize(image, (height, width), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        return image

    def process_image(self, image_path: pathlib.Path):
        if self.config.model in ["baseline", "none"]:
            return {}
        image = PIL.Image.open(image_path)
        if self.config.model in ['clip_momenta', 'clip', 'generic']:
            image = self.image_transform(image)
            if self.config.features_path:
                feature = self.image_features_loader(image_path)
                return {'image': image, 'image_feature': feature}
            else:
                return {'image': image}
        elif self.config.model == 'visualbert':
            image = self.image_transform(image_path)
        elif self.image_transform is not None:
            image = self.image_transform(image)
        return {'image': image}

    def process_text(self, text):
        if self.config.model == "ofa":
            if isinstance(text, str):
                text = text.lower().replace('-', ' ').replace('/', ' ')
                if not text:
                    return torch.tensor([], dtype=torch.int64)
                encoded = self.ofa_task.tgt_dict.encode_line(
                        line=self.ofa_task.bpe.encode(f" {text}"),
                        add_if_not_exist=False,
                        append_eos=False
                    ).long()
                return encoded[:self.config.max_text_len]
            elif isinstance(text, tuple):
                prefix, infix, suffix = tuple(map(self.process_text, text))
                total_length = prefix.shape[0] + infix.shape[0] + suffix.shape[0] + 2
                if total_length > self.config.max_text_len:
                    infix = infix[:max(total_length - self.config.max_text_len, 0)]
                encoded = torch.cat([self.ofa_bos, prefix, infix, suffix, self.ofa_eos])
                return encoded[:self.config.max_text_len]
            else:
                raise RuntimeError("text is of wrong type {type(text)}")
        elif self.config.model in ["clip", "clip_momenta", "generic"]:
            return self.tokenizer(text, truncate=True).squeeze(0)[:self.config.max_text_len]
        elif self.config.model in ["baseline", 'visualbert']:
            return self.tokenizer.encode_plus(
                    text,
                    max_length=self.config.max_text_len,
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"].squeeze(0).to(dtype=torch.long)
        else:
            return None

    def create_affirmation(self, entity: str, label: str) -> str:
        if label != "other":
            return f"{entity} is a {label}."
        else:
            assert(self.config.get("neither_for_other")) # For another commit
            classes = mmsrl.utils.LABELS.copy()
            classes.remove("other")
            self.rng.shuffle(classes)
            class_string = ", ".join(classes[:-1]) + f" nor {classes[-1]}"
            return f"{entity} is neither a {class_string}."

    def create_question(self, entity, ocr):
        classes = mmsrl.utils.LABELS.copy()
        self.rng.shuffle(classes)
        class_string = ", ".join(classes[:-1]) + f" or {classes[-1]}"
        #question = f"{entity} belongs to one of the following categorie: {class_string}. Which one is it ?"
        question = f"What is the category of {entity} between {class_string} ?"

        if self.config.vqa_input == "question ocr":
            return (question, ocr, "")
        elif self.config.vqa_input == "ocr question":
            return ("", ocr, question)
        elif self.config.vqa_input == "question":
            return ("", question, "")
        else:
            raise RuntimeError(f"Unknown value for config.vqa_input: {self.config.vqa_input}")

    def create_premises(self, entity, ocr):
        if self.config.get("ofa_classification_head"):
            labels = ["other"]
        else:
            labels = mmsrl.utils.LABELS

        premises = []
        for label in labels:
            affirmation = self.create_affirmation(entity, label)
            premises.append((('can image and text1 "', ocr, '" imply text2 " {affirmation} "?'), label))
        return premises

    def prepare_ofa_io(self, ocr, entity, true_label):
        test_answers: bool = self.is_eval and not self.config.get("ofa_classification_head")
        if self.config.get("ofa_classification_head"):
            true_label = "other"

        encoder_input = []
        decoder_input = []
        if self.config.ofa_task == "vqa_gen":
            source = self.create_question(entity, ocr)
            encoder_input.append(source)
            if test_answers:
                for label in mmsrl.utils.LABELS:
                    if self.config.get("decoder_input") is not None:
                        decoder_input.append((self.config.decoder_input.format(label=label), "", ""))
                    else:
                        decoder_input.append((source[0], source[1], f"{source[2]} {label}"))
            else:
                if self.config.get("decoder_input") is not None:
                    decoder_input.append((self.config.decoder_input.format(label=true_label), "", ""))
                else:
                    decoder_input.append((source[0], source[1], f"{source[2]} {true_label}"))
        elif self.config.ofa_task == "snli_ve":
            for source, label in self.create_premises(entity, ocr):
                encoder_input.append(source)
                if test_answers:
                    for answer in self.snli_answers:
                        if self.config.get("decoder_input") is not None:
                            decoder_input.append((self.config.decoder_input.format(label=answer), "", ""))
                        else:
                            decoder_input.append((source[0], source[1], f"{source[2]} {answer}"))
                else:
                    answer = "yes" if label == true_label else "no"
                    if self.config.get("decoder_input") is not None:
                        decoder_input.append((self.config.decoder_input.format(label=answer), "", ""))
                    else:
                        decoder_input.append((source[0], source[1], f"{source[2]} {answer}"))
        else:
            raise RuntimeError(f"Unknown value for self.config.ofa_task: {self.config.ofa_task}")
        return encoder_input, decoder_input

    def subsample_labels(self, entities, labels):
        if self.subsample_probs == [1, 1, 1, 1]:
            return entities, labels

        delete_me: List[int] = []
        for i, label in enumerate(labels):
            if self.config.get("multilabel"):
                for j, val in enumerate(label):
                    if val and torch.rand(1) > self.subsample_probs[j]:
                        label[j] = 0
                if label[j] == [0,0,0,0]:
                    delete_me.append(i)
            else:
                if torch.rand(1) > self.subsample_probs[label]:
                    delete_me.append(i)

        # We good a good score while doing this is old code…
        if self.config.get("keep_all_batches"):
            if len(delete_me) == len(entities):
                self.rng.shuffle(delete_me)
                delete_me = delete_me[1:]

        delete_me.sort(reverse=True)
        for index in delete_me:
            labels.pop(index)
            entities.pop(index)

        return entities, labels

    def generate_mask(self, input_ids, text):
        mask = torch.zeros(input_ids.shape[0]-1, dtype=torch.bool)
        mask[-2] = True
        if text[-1].endswith("victim") or text[-1].endswith("villain"):
            mask[-3] = True
        return mask

    def build_trie(self, answers: List[str]):
        self.trie = collections.defaultdict(lambda: torch.zeros(len(self.ofa_task.tgt_dict), dtype=torch.bool))
        for answer in answers:
            tokenized = torch.cat([self.process_text(answer), self.ofa_eos])
            for i in range(len(tokenized)):
                self.trie[tuple(tokenized[:i].tolist())][tokenized[i]] = True

    def get_constraint(self, target: torch.Tensor) -> torch.Tensor:
        constraint = torch.zeros((target.shape[0], len(self.ofa_task.tgt_dict)), dtype=torch.bool)
        for i in range(target.shape[0]):
            constraint[i] = self.trie[tuple(target[:i].tolist())]
        return constraint

    def index_to_sample(self, index):
        """return the input ids, images and target ids"""
        sample = {}

        sample["image_name"] = self.dataset[index]["image"]
        image_data = self.process_image(self.images_path / sample["image_name"])
        if image_data is None:
            return
        if self.config.model == "visualbert":
            sample['image_feature'] = image_data['image']
        else:
            sample.update(image_data)
        
        sample["raw_text"] = self.dataset[index]["OCR"]
        if self.config.get('use_caption'):
            caption = self.captions_df[self.captions_df[0]==sample["image_name"]][1].item() + '. '
            sample["caption"] = caption
            sample["caption_input_ids"] = self.process_text(sample["caption"])

        if self.has_labels:
            targets = {l: self.dataset[index][l] for l in mmsrl.utils.LABELS}
            entities = [t for l in targets for t in targets[l]]

            if self.config.get("multilabel"):
                labels = []
                for ent in entities:
                    ent_lab = [mmsrl.utils.LABELS.index(l) for l in targets for t in targets[l] if t==ent]
                    ent_lab_distrib = [0]*len(mmsrl.utils.LABELS)
                    for l in ent_lab: ent_lab_distrib[l] = 1
                    labels.append(ent_lab_distrib)
            else:
                labels = [mmsrl.utils.LABELS.index(l) for l in targets for t in targets[l]]

            if not self.is_eval:
                entities, labels = self.subsample_labels(entities, labels)
                if not entities:
                    return

                ids = list(range(len(entities)))
                self.rng.shuffle(ids)
                ids = ids[:self.config.get("max_nb_ent", len(ids))]
                entities = [entities[i] for i in ids]
                labels = [labels[i] for i in ids]

            sample["labels"] = torch.tensor(labels, dtype=torch.int64)
        else:
            entities = self.dataset[index]["entity_list"]

        sample["entities_name"] = entities

        if self.config.model == "ofa":
            if self.config.get("group_image_entities"):
                assert(self.config.get("ofa_classification_head"))
                if self.has_labels:
                    sample["labels"] = torch.tensor(labels, dtype=torch.int64)
                else:
                    None
                sample["text"] = []
                sample["decoder_text"] = []
                for i, entity in enumerate(entities):
                    label = labels[i] if self.has_labels else None
                    label_text = mmsrl.utils.LABELS[label] if label is not None else None
                    encoder_input, decoder_input = self.prepare_ofa_io(sample["raw_text"], entity, label_text)
                    assert(len(encoder_input) == 1)
                    assert(len(decoder_input) == 1)
                    sample["text"].append(encoder_input[0])
                    sample["decoder_text"].append(decoder_input[0])
                sample["input_ids"] = list(map(self.process_text, sample["text"]))
                decoder_text = list(map(self.process_text, sample["decoder_text"]))
                if not self.config.get("ofa_classification_head") and not self.config.get("keep_eof"):
                    decoder_text = [text[:-1] for text in decoder_text]
                sample["decoder_input_ids"] = [text[:-1] for text in decoder_text]
                sample["decoder_target"] = [text[1:] for text in decoder_text]
                if self.config.get("constrain_output"):
                    sample["decoder_constraint"] = [self.get_constraint(target) for target in sample["decoder_target"]]
                yield sample
            else:
                for i, entity in enumerate(entities):
                    current_sample = {}
                    current_sample["raw_text"] = sample["raw_text"]
                    current_sample["image_name"] = sample["image_name"]
                    current_sample["image"] = sample["image"]

                    if self.has_labels:
                        label = labels[i]
                        current_sample["labels"] = torch.tensor([label], dtype=torch.int64)
                    else:
                        label = None

                    label_text = mmsrl.utils.LABELS[label] if label is not None else None
                    encoder_input, decoder_input = self.prepare_ofa_io(sample["raw_text"], entity, label_text)
                    current_sample["text"] = encoder_input
                    current_sample["decoder_text"] = decoder_input

                    current_sample["input_ids"] = list(map(self.process_text, encoder_input))
                    decoder_text = list(map(self.process_text, decoder_input))
                    if not self.config.get("ofa_classification_head") and not self.config.get("keep_eof"):
                        decoder_text = [text[:-1] for text in decoder_text]
                    current_sample["decoder_input_ids"] = [text[:-1] for text in decoder_text]
                    current_sample["decoder_target"] = [text[1:] for text in decoder_text]
                    if self.config.get("only_score_answer") or self.config.get("only_train_answer"):
                        current_sample["decoder_answer_mask"] = [self.generate_mask(input_ids, text) for input_ids, text in zip(decoder_text, decoder_input)]
                    if self.config.get("constrain_output"):
                        current_sample["decoder_constraint"] = [self.get_constraint(target) for target in current_sample["decoder_target"]]

                    current_sample["entities_name"] = [entity]
                    yield current_sample
        else:
            sample["text"] = sample["raw_text"]
            sample["input_ids"] = self.process_text(sample["text"])
            if self.config.get('use_visualbert'):
                sample["vbert_input_ids"] = self.vbert_tokenizer.encode_plus(
                    sample["text"],
                    max_length=self.config.max_text_len,
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"].squeeze(0).to(dtype=torch.long)
            sample["entities_input_ids"] = [self.process_text(entity) for entity in entities]
            yield sample

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_modulo: int = 1
            worker_residue: int = 0
        else:
            worker_modulo: int = worker_info.num_workers
            worker_residue: int = worker_info.id

        if not self.is_eval and self.config.get("batch_per_epoch"):
            sample_per_worker = len(self) // worker_modulo + (1 if worker_residue < len(self) % worker_modulo else 0)
            sample_generated = 0
            while sample_generated < sample_per_worker:
                samples = list(self.index_to_sample(self.rng.randint(0, len(self.dataset)-1)))
                yield from samples[:sample_per_worker - sample_generated]
                sample_generated += len(samples)
        else:
            self.order = list(range(len(self.dataset)))
            if not self.is_eval:
                self.common_rng.shuffle(self.order)

            for i, index in enumerate(self.order):
                if i % worker_modulo == worker_residue:
                    yield from self.index_to_sample(index)
