from typing import Any, Dict, List, Set, Tuple, Union
import itertools
import os
import pathlib
import random

import IPython.display
import ipywidgets

import mmsrl.dataset
import mmsrl.utils


SAMPLE_PER_LABEL = 25 # Total of 100 annotations
SEED = 0 # We don't want to cherry pick what we annotate…
#PREFIX = pathlib.Path(os.environ["DATA_PATH"]) / "mmsrl" # we can put annotations in data
PREFIX = pathlib.Path(".") # The current directory (annotations in the git repository) seems better (we can commit the annotations)
SELECTED_PATH = PREFIX / "selected for annotation"


def get_data() -> Tuple[Dict[str, Any], List[Set[Tuple[str, str]]]]:
    """
    Get the sample needing annotations split into the 4 labels.
    
    Returns:
        - the dataset, a image→sample dictionary
        - sets of (image, entity) pairs grouped by label
    """
    datasets = {}
    for split in ["train", "val", "test"]:
        config = mmsrl.utils.dotdict(dataname="all", model="none")
        datasets[split] = mmsrl.dataset.MemeDataset(config, split, is_eval=True)
    data = {sample["image_name"]: sample for sample in itertools.chain(*datasets.values())}
    annotable = [set() for _ in range(4)]
    for image, sample in data.items():
        for entity, label in zip(sample["entities_name"], sample["labels"]):
            annotable[label].add((image, entity))
    return data, annotable


def select_annotations(annotable: List[Set[Tuple[str, str]]]):
    """ Write the keys to the selected file if it does not exist. """
    if SELECTED_PATH.exists():
        raise RuntimeError("The file containing selected samples already exists, some people might have started annotation, don't change it now!")
    rng = random.Random(SEED)
    selected: List[Tuple[str, str]] = []
    for label in annotable:
        domain: List[Tuple[str, str]] = sorted(label) # to ensure determinism
        selected.extend(rng.sample(domain, SAMPLE_PER_LABEL))
    rng.shuffle(selected) # we don't want to present all victim in a row, etc.
    with SELECTED_PATH.open("w") as selected_file:
        for image, entity in selected:
            print(f"{image}\t{entity}", file=selected_file)


class Annotator:
    def __init__(self, identity: str, image_prefix: Union[pathlib.Path, str]):
        self.identity = identity
        self.image_prefix = pathlib.Path(image_prefix)
        self.annotated_path = PREFIX / f"annotations of {self.identity}"
        self.data, _ = get_data()

    def instructions(self):
        IPython.display.display(ipywidgets.HTML(f"""
            <p><strong>This notebook is for annotator {self.identity}!</strong></p>
            <p><strong>Instructions:</strong></p>
            <ul>
                <li>Given a meme and an entity, determine the role of the entity in the meme: hero vs. villain vs. victim vs. other.</li>
                <li>The meme is to be analyzed from the perspective of the author of the meme.</li>
                <li>Class definitions:<dl>
                    <dt>Hero</dt>
                    <dd>The entity is presented in a positive light. Glorified for their actions conveyed via the meme or gathered from background context.</dd>
                    <dt>Villain</td>
                    <dd>The entity is portrayed negatively, e.g., in an association with adverse traits like wickedness, cruelty, hypocrisy, etc.</dd>
                    <dt>Victim</dt>
                    <dd>The entity is portrayed as suffering the negative impact of someone else’s actions or conveyed implicitly within the meme.</dd>
                    <dt>Other</dt>
                    <dd>The entity is not a hero, a villain, or a victim.</dd>
                    </dl></li>
                <li>The four classes are balanced, you should give each answer roughly 25% of the time.</li>
                <li>Your answers are saved at each annotation, you can leave and come back at any time.</li>
            </ul>
            """))

    def get_already_annotated(self) -> Set[Tuple[str, str]]:
        """ Get list of samples already annotated. """
        annotated = set()
        try:
            with self.annotated_path.open("r") as annotated_file:
                for line in annotated_file:
                    image, entity, _ = line.rstrip('\n').split('\t')
                    annotated.add((image, entity))
        except FileNotFoundError:
            pass
        return annotated

    def get_next_annotation(self) -> Tuple[str, str]:
        """ Get the next (image, entity) pair to annotate. """
        # We re-read the file for each annotation to avoid an error-prone in-memory updating of the list
        annotated = self.get_already_annotated()
        # It would be hard to be less algorithmicaly efficient, but better be safe than sorry
        with SELECTED_PATH.open("r") as selected_file:
            for i, line in enumerate(selected_file):
                image, entity = line.rstrip('\n').split('\t')
                if (image, entity) not in annotated:
                    return i, image, entity
        return -1, None, None

    def write_answer(self, image: str, entity: str, answer: int):
        with self.annotated_path.open("a") as annotated_file:
            print(f"{image}\t{entity}\t{answer}", file=annotated_file)

    def get_sample_html(self, image: str, entity: str):
        sample = self.data[image]
        ocr = sample["raw_text"]
        html = f"""
        <img src="{self.image_prefix / image}"/>
        <pre>{ocr}</pre>
        """
        return ipywidgets.HTML(html)

    def delete_last_annotation(self):
        annotations = []
        try:
            with self.annotated_path.open("r") as annotated_file:
                annotations = annotated_file.readlines()
        except FileNotFoundError:
            pass
        if annotations:
            annotations.pop()
        with self.annotated_path.open("w") as annotated_file:
            annotated_file.writelines(annotations)

    def answer_clicked(self, image: str, entity: str, label: int):
        def callback(button):
            IPython.display.clear_output()
            if label == -1:
                # Undo last answer
                self.delete_last_annotation()
            else:
                assert(button.description == mmsrl.utils.LABELS[label])
                self.write_answer(image, entity, label)
            self.ask_next()
        return callback

    def ask_annotation(self, current_id: int, image: str, entity: str):
        sample = self.get_sample_html(image, entity)
        buttons = []
        for ilabel, label in enumerate(mmsrl.utils.LABELS):
            button = ipywidgets.Button(description=label)
            button.on_click(self.answer_clicked(image, entity, ilabel))
            buttons.append(button)
        button = ipywidgets.Button(description="Undo last answer")
        button.on_click(self.answer_clicked(None, None, -1))
        buttons.append(button)
        hbox = ipywidgets.HBox(buttons)
        instructions = ipywidgets.HTML(f'({current_id+1}/{4*SAMPLE_PER_LABEL}) <strong>What is the role of <span style="font-size:150%;">{entity}</span> in the above meme?</strong>')
        vbox = ipywidgets.VBox([sample, instructions, hbox])
        IPython.display.display(vbox)

    def completed(self):
        message = ipywidgets.HTML(f'You completed all annotations! Thank you!')
        button = ipywidgets.Button(description="Undo last answer")
        button.on_click(self.answer_clicked(None, None, -1))
        vbox = ipywidgets.VBox([message, button])
        IPython.display.display(vbox)

    def ask_next(self):
        current_id, image, entity = self.get_next_annotation()
        if current_id < 0 and image is None and entity is None:
            self.completed()
        else:
            self.ask_annotation(current_id, image, entity)


# This only need to be done once for all annotators
# And it was already done, DO NOT UNCOMMENT
# select_annotations(get_data()[1])
