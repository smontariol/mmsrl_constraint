from typing import Any, Dict, Optional, Type
import pathlib
import pickle
import types

import torch

import mmsrl.utils


class Output:
    """
    Class for outputing data about the learned model.
    """
    # TODO parametrize this class

    def __init__(self, config: mmsrl.utils.dotdict, split: str, epoch: int, final: bool) -> None:
        """ Initialize the outputs variables, but do not acquire necessary resources yet. """
        self.config: mmsrl.utils.dotdict = config
        self.split = {"unseen_test": "test"}.get(split, split)
        self.epoch = epoch
        self.final = final
        self.path = None
        self.file = None

    def __enter__(self):
        """ Acquire resources needed for outputing the data. """
        if self.final and self.config.get(f"output_{self.split}"):
            self.path = pathlib.Path(self.config[f"output_{self.split}"])
            if '{' in str(self.path):
                self.path = pathlib.Path(str(path).format(self.epoch))
            self.file = self.path.open("wb")
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_inst: Optional[BaseException],
                 exc_tb: Optional[types.TracebackType]) -> None:
        """ Free used resources. """
        if self.file is not None:
            self.file.close()

    def __call__(self, batch: Dict[str, Any], predictions: Optional[torch.Tensor]) -> None:
        """
        Update model output data files with the given batch and the outputs of the model on this batch.

        Args:
            batch: the input values used for evaluation
            predictions: the output of the model
        """
        if self.file and predictions is not None:
            for ib in range(len(batch["image_name"])):
                entities = dict(zip(
                    filter(None, batch["entities_name"][ib]),
                    predictions[ib].detach().cpu()))
                pickle.dump((batch["image_name"][ib], entities), self.file)
                self.file.flush()
