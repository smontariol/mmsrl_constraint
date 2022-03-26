import argparse
import contextlib
import copy
import logging
import math
import multiprocessing
import pathlib
import signal
import sys
import tqdm
import types

# Do this do deal with the badly packaged OFA
import mmsrl
sys.path.append(mmsrl.__path__[0] + "/../OFA")

import clip
import fairseq.checkpoint_utils
import fairseq.dataclass.utils
import fairseq.options
import fairseq.tasks
import fairseq.utils
import sklearn.metrics
import tasks # OFA
import torch
import transformers

from mmsrl.dataset import MemeDataset
import mmsrl.batcher
import mmsrl.extract_image_features
import mmsrl.output
import mmsrl.utils


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.prepare_framework()
        self.prepare_data()
        self.prepare_model()

    def hook_signals(self) -> None:
        """ Change the behavior of SIGINT (^C) to change a variable `self.interrupted' before killing the process. """
        self.interrupted = False

        def handler(sig: int, frame: types.FrameType) -> None:
            if multiprocessing.current_process().name != "MainProcess":
                return

            print("\n\033[31mInterrupted, execution will stop at the next end of validation.\n\033[1mNEXT ^C WILL KILL THE PROCESS!\033[0m\n", file=sys.stderr)
            self.interrupted = True
            signal.signal(signal.SIGINT, signal.SIG_DFL)

        signal.signal(signal.SIGINT, handler)

    def get_ofa_config(self):
        parser = fairseq.options.get_generation_parser()
        input_args = ["",
                    f"--task={self.config.ofa_task}",
                    "--unnormalized",
                    f"--path={self.config.ofa_path}",
                    "--bpe-dir=OFA/utils/BPE"]
        args = fairseq.options.parse_args_and_arch(parser, input_args)
        ofa_config = fairseq.dataclass.utils.convert_namespace_to_omegaconf(args)
        return ofa_config

    def prepare_feature(self, feature):
        """ Adapt image and text feature to the model being in float16 or float32 setting. """
        if isinstance(feature, torch.Tensor):
            if self.config.get("half"):
                feature = feature.to(dtype=torch.float16)
            feature = feature.to(device=self.device)
        return feature

    def prepare_framework(self):
        if self.config.model == "ofa":
            if self.config.ofa_task == 'vqa_gen':
                fairseq.tasks.register_task('vqa_gen', tasks.mm_tasks.vqa_gen.VqaGenTask)
            elif self.config.ofa_task == 'snli_ve':
                fairseq.tasks.register_task('snli_ve', tasks.mm_tasks.snli_ve.SnliVeTask)
            else:
                raise RuntimeError(f"Unknown OFA task {self.config.ofa_task}")
            ofa_pre_config = self.get_ofa_config()
            self.ofa_task = fairseq.tasks.setup_task(ofa_pre_config.task)
            self.ofa_models, self.ofa_config = fairseq.checkpoint_utils.load_model_ensemble(
                    fairseq.utils.split_paths(ofa_pre_config.common_eval.path),
                    task=self.ofa_task
                )
        elif self.config.model in ["clip", "clip_momenta", "generic"]:
            self.clip_model, self.clip_preprocess = clip.load(self.config.modelname, device=self.device)

    def prepare_data(self):
        self.epoch = 0
        self.dataset = {}
        self.iterator = {}
        if self.config.model == "ofa":
            extra_dataset_kwargs = {"ofa_task": self.ofa_task}
            text_pad: int = self.ofa_task.src_dict.pad() 
        elif self.config.model == "clip":
            text_pad: int = 0
            if self.config.features_path:
                self.image_feature_extractor = mmsrl.extract_image_features.ImageFeatureExtractor(self.config)
                extra_dataset_kwargs = {"image_processor": self.clip_preprocess, "tokenizer": clip.tokenize, "image_features_loader": self.image_feature_extractor.process_image}
            else:
                extra_dataset_kwargs = {"image_processor": self.clip_preprocess, "tokenizer": clip.tokenize}
        elif self.config.model == "clip_momenta":
            text_pad: int = 0
            self.image_feature_extractor = mmsrl.extract_image_features.ImageFeatureExtractor(self.config)
            extra_dataset_kwargs = {"image_processor": self.clip_preprocess, "tokenizer": clip.tokenize, "image_features_loader": self.image_feature_extractor.process_image}
        elif self.config.model == "visualbert":
            self.image_feature_extractor = mmsrl.extract_image_features.ImageFeatureExtractor(self.config)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.modelname)
            text_pad: int = self.tokenizer.pad_token_id
            extra_dataset_kwargs = {"image_processor": self.image_feature_extractor.process_image, "tokenizer": self.tokenizer}
        elif self.config.model == "generic":
            # tokenizer:
            extra_dataset_kwargs = {"tokenizer": clip.tokenize, "image_processor": self.clip_preprocess}
            text_pad: int = 0
            if self.config.use_visualbert or self.config.features_path:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.visualbert_text_model)
                self.image_feature_extractor = mmsrl.extract_image_features.ImageFeatureExtractor(self.config)
                extra_dataset_kwargs.update({"image_features_loader": self.image_feature_extractor.process_image, "vbert_tokenizer": self.tokenizer})
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.modelname)
            text_pad: int = self.tokenizer.pad_token_id
            extra_dataset_kwargs = {"tokenizer": self.tokenizer}

        self.batcher = mmsrl.batcher.Batcher(self.config, text_pad=text_pad, label_pad=-100, image_pad=0)

        for split in ["train", "val", "test"]:
            self.dataset[split] = MemeDataset(self.config, split, split!="train", **extra_dataset_kwargs)
            self.iterator[split] = lambda split=split, trainer=self: tqdm.tqdm(
                    torch.utils.data.DataLoader(dataset=trainer.dataset[split],
                                                batch_size=trainer.config["batch_size"],
                                                collate_fn=trainer.batcher,
                                                num_workers=trainer.config.get("workers", 0),
                                                worker_init_fn=trainer.dataset[split].seed,
                                                ), desc=f"{split}, epoch: {self.epoch}", leave=False)

    def prepare_model(self) -> None:
        label_weights = self.dataset['train'].get_label_weights()
        if self.config.model=="ofa":
            from mmsrl.ofa import Model
            model_kwargs = {"ofa_models": self.ofa_models, "ofa_config": self.ofa_config, "ofa_task": self.ofa_task}
        elif self.config.model=="perceiver":
            from mmsrl.perceiver import Model
            model_kwargs = {}
        elif self.config.model=="clip":
            from mmsrl.clip import Model
            model_kwargs = {"model": self.clip_model}
        elif self.config.model=="clip_momenta":
            from mmsrl.clip_momenta import Model
            model_kwargs = {"model": self.clip_model}
        elif self.config.model == "visualbert":
            from mmsrl.visualbert import Model
            model_kwargs = {}
        elif self.config.model == "generic":
            from mmsrl.generic_classifier import Model
            model_kwargs = {"model": self.clip_model}
        else:
            from mmsrl.baseline import Model
            model_kwargs = {}
        self.model = Model(self.config, label_weights.to(device=self.device), **model_kwargs)
        if self.config.get("half"):
            self.model.half()
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                            lr=self.config["learning_rate"],
                                            weight_decay=self.config.get("weight_decay", 0))
        step_per_epoch = math.ceil(len(self.dataset["train"]) / self.config.batch_size / self.config.accumulation)
        max_iterations: int = self.config["max_epoch"] * step_per_epoch

        warmup_step = self.config.get("warmup_step", 0)
        warmup_step = warmup_step * max_iterations if 0 < warmup_step < 1 else warmup_step
        self.scheduler = transformers.get_scheduler(self.config.get("lr_scheduler", "linear"), self.optimizer, warmup_step, max_iterations)
        self.scaler = torch.cuda.amp.GradScaler(1)  # Used to scale gradients to avoid rounding to 0 when using float16
        self.amp = torch.cuda.amp.autocast if self.config.amp else contextlib.nullcontext

    def eval(self, split: str, final: bool = False) -> float:
        with torch.no_grad(), mmsrl.output.Output(self.config, split, self.epoch, final) as output:
            predictions = []
            truth = []
            self.model.eval()
            accuracy_accumulator = 0
            num_samples = 0
            loop = self.iterator[split]()
            for batch in loop:
                batch = {key: self.prepare_feature(value) for key, value in batch.items()}
                with self.amp():  # Use mixed-precision float16
                    loss, prediction = self.model(**batch)
                    output(batch, prediction)
                if prediction is not None and "labels" in batch:
                    predictions.extend(prediction.argmax(2)[batch['entities_mask']].tolist())
                    truth.extend(batch["labels"][batch['entities_mask']].tolist())
                    accuracy_accumulator += ((prediction.argmax(2) == batch["labels"])[batch['entities_mask']]).sum().item()
                    num_samples += batch["entities_mask"].sum().item()
                    loop.set_postfix(accuracy=f"{accuracy_accumulator/num_samples:5.3f}") # for the tqdm
            if predictions and truth:
                f1 = sklearn.metrics.f1_score(truth, predictions, average="macro")
                class_precision, class_recall, class_f1, _ = sklearn.metrics.precision_recall_fscore_support(truth, predictions)
                class_scores = ", ".join(f"{label}[F{class_f1[i]:4.2f}/P{class_precision[i]:4.2f}/R{class_recall[i]:4.2f}]" for i, label in enumerate(mmsrl.utils.LABELS))
                print(f"Epoch {self.epoch:3} {split}, Macro-F1: {f1:5.3f}, {class_scores}")
                return f1
            return None

    def train_step(self) -> None:
        """ Apply the gradients to the parameters. """
        if self.config.amp:
            self.scaler.unscale_(self.optimizer)  # Gradients are scaled in order to avoid rounding to 0 when casting to float16
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])  # Clip to gradients to a maximum value (to be robust to outliers)
        if self.config.amp:
            self.scaler.step(self.optimizer)  # Add the gradient to the parameters
        else:
            self.optimizer.step()
        self.scheduler.step()  # Update the learning rate
        if self.config.amp:
            self.scaler.update()  # Update by how much the gradients should be scaled to avoid rounding to 0
        self.optimizer.zero_grad()  # Set the gradients to 0 to prepare for the next iteration

    def train(self) -> None:
        best_dev = -math.inf
        best_dev_epoch = 0
        best_state_dict = None  # Used to save the best model
        metric_key = None

        self.hook_signals()
        if not self.config.get("no_initial_valid"):
            best_dev = self.eval("val", final=True)

        for self.epoch in range(1, 1 + self.config["max_epoch"]):
            # Accumulate gradient to better estimate it even with small batches
            if self.interrupted == True:
                break
            metric_accumulator = 0
            num_samples = 0
            self.model.train()
            epoch_loop = self.iterator["train"]()
            for batch_id, batch in enumerate(epoch_loop):
                if self.interrupted == True:
                    break
                if batch_id >= self.config.get("batch_per_epoch", math.inf):
                    break
                batch = {key: self.prepare_feature(value) for key, value in batch.items()}
                with self.amp():  # Use mixed-precision float16
                    loss, prediction = self.model(**batch)
                    if loss is None:
                        break

                if prediction is not None:
                    metric_accumulator += ((prediction.argmax(2) == batch["labels"])[batch['entities_mask']]).sum().item()
                    metric_key = "accuracy"
                elif loss is not None:
                    metric_accumulator += loss * batch["entities_mask"].sum().item()
                    metric_key = "loss"
                num_samples += batch["entities_mask"].sum().item()
                if metric_key is not None:
                    epoch_loop.set_postfix(**{metric_key: f"{metric_accumulator/num_samples:5.3f}"}) # for the tqdm

                self.scaler.scale(loss).backward()  # Accumulate the (scaled) gradients
                if (1+batch_id) % self.config.get("accumulation", 1) == 0:
                    self.train_step()

            if metric_key is not None:
                print(f"Epoch {self.epoch:3} train, {metric_key}: {metric_accumulator/num_samples:.4f}")
            else:
                print(f"Epoch {self.epoch:3} train, done")
            if loss is None:
                break

            if (batch_id+1) % self.config.get("accumulation", 1) != 0:
                self.train_step()

            dev = self.eval("val")
            if dev > best_dev:  # If dev score improved
                best_dev = dev
                best_dev_epoch = self.epoch
                if self.config.get("save_model"):
                    torch.save(self.model.state_dict(), self.config.save_model)
                    best_state_dict = True
                else:
                    best_state_dict = copy.deepcopy(self.model.state_dict())  # Save the weights of the model
            elif self.epoch - best_dev_epoch > self.config.get("patience", 0):  # If dev score worsened for several steps (or 1 if patience is 0)
                break # Early stopping

            self.dataset["train"].compute_subsample(self.epoch)
        if best_state_dict is not None:  # If we improved over random initialization
            if self.config.get("save_model"):
                self.model.load_state_dict(torch.load(self.config.save_model))
            else:
                self.model.load_state_dict(best_state_dict)  # Load the model with the best dev score
        for split in ["val", "test"]:
            self.eval(split, final=True)


if __name__=="__main__":
    config: mmsrl.utils.dotdict = mmsrl.utils.parse_args()
    print("\033[33mConfiguration:\033[0m")
    mmsrl.utils.print_dict(config)
    Trainer(config).train()
