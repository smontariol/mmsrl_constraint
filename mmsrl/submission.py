from typing import Any, Dict
import argparse
import collections
import json
import os
import pathlib
import pickle
import zipfile

import mmsrl.utils


def make_submission_file(config: mmsrl.utils.dotdict) -> None:
    inputs: Dict[str, Dict[str, Any]] = collections.defaultdict(dict)
    with config.prediction.open("rb") as input_file:
        try:
            while True:
                image, entities = pickle.load(input_file)
                for entity, prediction in entities.items():
                    inputs[image][entity] = prediction
        except EOFError:
            pass

    data_path = pathlib.Path(os.environ['DATA_PATH'])
    annot_file = data_path / "mmsrl" / "all" / "annotations" / f"unseen_test.jsonl"
    with annot_file.open('r') as json_file:
        dataset = list(map(json.loads, json_file))

    outputfile_path = config.output /  pathlib.Path(config.prediction.stem).with_suffix('.json')
    with outputfile_path.open("w") as output_file:
        for sample in dataset:
            image = sample["image"]
            output: Dict[str, Any] = {"image": image}
            for label in mmsrl.utils.LABELS:
                output[label] = []

            for i, entity in enumerate(sample["entity_list"]):
                if entity == "":
                    prediction = mmsrl.utils.LABELS.index("other")
                else:
                    predictions = [(p, i) for i, p in enumerate(inputs[image][entity].tolist())]
                    predictions.sort(reverse=True)
                    if len(set(entities)) == len(entities) or entities.count(entity) == 1:
                        prediction = predictions[0][1]
                    else:
                        prediction = predictions[entities[:i].count(entity)][1]
                output[mmsrl.utils.LABELS[prediction]].append(entity)

            print(json.dumps(output), file=output_file)

    with zipfile.ZipFile(str(outputfile_path.with_suffix('.zip')), mode="w") as archive:
        archive.write(str(outputfile_path), arcname="answer.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path, help="Path were we write the submission json file.")
    parser.add_argument("prediction", type=pathlib.Path, help="Prediction file to use.")
    args = mmsrl.utils.dotdict(vars(parser.parse_args()))
    make_submission_file(args)
