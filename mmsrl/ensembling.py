from typing import Any, Dict, List, Tuple
import argparse
import collections
import json
import os
import re
import sys
import pathlib
import pickle

import tqdm
import numpy
import sklearn.metrics
import sklearn.linear_model
import xgboost

import mmsrl.utils


def build_model_paths(paths):
    model_paths = set()
    for path in paths:
        for filename in os.listdir(path):
            modelname = filename.replace("val", "{}").replace("test", "{}")
            if modelname.count("{}") == 1:
                model_paths.add(path / modelname)
            elif modelname.count("{}") > 1:
                print(f"\033[31;1mWARNING: ignoring file with several \"val\" and \"test\" in filename: {filename}.\033[0m")
    return model_paths


def read_prediction_file(path):
    try:
        data = collections.defaultdict(dict)
        with open(path, "rb") as file:
            try:
                while True:
                    image, entities = pickle.load(file)
                    for entity, prediction in entities.items():
                        data[image][entity] = prediction.numpy()
            except EOFError:
                return data
    except FileNotFoundError:
        return {}


def read_predictions(paths):
    data = {}
    for path in tqdm.tqdm(paths, desc="Loading predictions", leave=False):
        prediction = str(path)
        data[prediction] = {}
        for split in ["val", "test"]:
            path = prediction.replace("{}", split)
            data[prediction][split] = read_prediction_file(path)
    return data


def read_data(split):
    data_path = pathlib.Path(os.environ["DATA_PATH"])
    path = data_path / "mmsrl" / "all" / "annotations" / f"{split}.jsonl"
    with path.open("r") as file:
        return list(map(json.loads, file))


def write_prediction_file(path, data):
    with open(path, "wb") as file:
        for image, predictions in data.items():
            pickle.dump((image, predictions), file)


def get_sample_entities(sample):
    if "entity_list" in sample:
        return sample["entity_list"]
    else:
        return sum((sample[label] for label in mmsrl.utils.LABELS), [])


def models_to_tensors(data, models):
    tensors = {}
    for sample in data:
        image = sample["image"]
        tensors[image] = {}
        for entity in set(get_sample_entities(sample)):
            if not entity:
                continue
            values = []
            for model in models:
                values.append(model[image][entity])
            tensors[image][entity] = numpy.stack(values)
    return tensors


def model_is_valid(model, dataset):
    try: 
        for sample in dataset:
            image = sample["image"]
            for entity in get_sample_entities(sample):
                if entity:
                    pred = model[image][entity]
                    if not numpy.isfinite(pred).all():
                        return False
    except KeyError:
        return False
    return True


def compute_score(predictions, truth, final, split):
    y_pred = []
    y_truth = []
    for sample in truth:
        image = sample["image"]
        for ilabel, label in enumerate(mmsrl.utils.LABELS):
            for entity in sample[label]:
                if entity:
                    y_pred.append(predictions[image][entity].argmax().item())
                else:
                    y_pred.append(3)
                y_truth.append(ilabel)

    f1 = sklearn.metrics.f1_score(y_truth, y_pred, average="macro", zero_division=0)
    class_precision, class_recall, class_f1, _ = sklearn.metrics.precision_recall_fscore_support(y_truth, y_pred, zero_division=0)
    class_scores = ", ".join(f"{label}[F{class_f1[i]:4.2f}/P{class_precision[i]:4.2f}/R{class_recall[i]:4.2f}]" for i, label in enumerate(mmsrl.utils.LABELS))
    color = "32" if final else "33"
    return f1, f"{split} Macro-F1: \033[{color};1m{f1:5.3f}\033[0m, {class_scores}"


def models_to_xy(models, valid):
    nmodels = len(models)
    x = []
    y = []
    order = []
    for sample in valid:
        image = sample["image"]
        for ilabel, label in enumerate(mmsrl.utils.LABELS):
            for entity in sample[label]:
                if not entity:
                    continue
                row = numpy.empty((nmodels,4), dtype=numpy.float32)
                for im, model in enumerate(models):
                    row[im] = model[image][entity]
                x.append(row)
                y.append(ilabel)
                order.append((image, entity))
    return numpy.stack(x), numpy.array(y, dtype=numpy.int64), order


def build_ensemble(config, valid, models):
    tmodels = models_to_tensors(valid, models)
    if config.aggregator in ["linear", "xgboost"]:
        x, y, order = models_to_xy(models, valid)

    if config.aggregator in "linear":
        if config.full_matrix:
            linear = sklearn.linear_model.LogisticRegression(penalty="none", class_weight="balanced", max_iter=1000)
            linear.fit(x.reshape(x.shape[0], -1), y)
        else:
            nsample = y.shape[0]
            label_weight = nsample / 4 / numpy.bincount(y)
            nmodels = len(models)
            x = x.transpose(0, 2, 1)
            x = x.reshape(-1, nmodels)
            y = y.repeat(4)
            sample_weight = label_weight[y]
            y = (y.reshape(nsample, 4) == numpy.arange(4)).flatten()

            linear = sklearn.linear_model.LogisticRegression(penalty="none", max_iter=1000)
            linear.fit(x, y, sample_weight=sample_weight)
        ensemble = linear.coef_
    elif config.aggregator == "xgboost":
        nsample = y.shape[0]
        label_weight = nsample / 4 / numpy.bincount(y)
        param = {'objective': 'multi:softmax', 'num_class': 4}
        param.update(eval(config.xgboost_params))
        print(f"\033[33mXGBoost parameters: {param}\033[0m")
        ensemble = xgboost.XGBRegressor(**param)
        ensemble.fit(x.reshape(2067, -1), y, sample_weight=label_weight[y])
        if config.save_xgboost:
            ensemble.save_model(str(config.output / get_output_name(config)) + ".xgboost.json")
            with open(str(config.output / get_output_name(config)) + ".xgboost.map", "wb") as outfile:
                pickle.dump(order, outfile)
    else:
        ensemble = None

    prediction = ensemble_predict(config, valid, ensemble, models)
    print(compute_score(prediction, valid, True, "VALID")[1])
    return ensemble


def ensemble_predict_sample(config, ensemble, models):
    if config.aggregator == "mean":
        return models.mean(0)
    elif config.aggregator == "max":
        return models.max(0)[0]
    elif config.aggregator == "min":
        return models.min(0)[0]
    elif config.aggregator == "median":
        models.sort(0)
        return models[models.shape[0]//2, :]
    elif config.aggregator == "linear":
        return ensemble.dot(models.flatten() if config.full_matrix else models)
    elif config.aggregator == "xgboost":
        probabilities = numpy.zeros(4, dtype=numpy.float32)
        prediction = ensemble.predict(models.reshape(1, -1))
        probabilities[int(prediction)] = 1
        return probabilities
    else:
        raise RuntimeError(f"Unknown aggregator {config.aggregator}")


def ensemble_predict(config, test, ensemble, models):
    tmodels = models_to_tensors(test, models)
    output = {}
    for sample in test:
        image = sample["image"]
        output[image] = {}
        for entity in set(get_sample_entities(sample)):
            if entity:
                output[image][entity] = ensemble_predict_sample(config, ensemble, tmodels[image][entity])
            else:
                output[image][entity] = numpy.array([0, 0, 0, 1], dtype=numpy.float32)
    return output


def filter_models(config: mmsrl.utils.dotdict, models, val_data, test_data):
    useme = {}
    for model_name, model in models.items():
        if not (model_is_valid(model["val"], val_data) and model_is_valid(model["test"], test_data)):
            continue
        f1, msg = compute_score(model["val"], val_data, False, "VALID")
        if f1 > config.cut_off:
            useme[model_name] = model
            if config.show_all:
                print(f"\033[33m{model_name}\033[0m")
                print(msg)
    return useme


def get_output_name(config: mmsrl.utils.dotdict) -> str:
    name = f"ensemble_{config.aggregator}_"
    name += str(config.cut_off).split('.')[-1]
    if config.aggregator == "linear" and config.full_matrix:
        name += "_fullmatrix"
    if config.aggregator == "xgboost":
        name += re.sub("[^a-zA-Z0-9.]+", "_", config.xgboost_params).rstrip('_')
    return f"{name}.pkl"


def perform_ensembling(config: mmsrl.utils.dotdict, models, val_data, test_data):
    models_filtered = filter_models(config, models, val_data, test_data)
    models_order = list(models_filtered.keys())
    models = [models[key] for key in models_order]
    ensemble = build_ensemble(config, val_data, [model["val"] for model in models])
    prediction = ensemble_predict(config, test_data, ensemble, [model["test"] for model in models])
    outfile = config.output / get_output_name(config)
    print(f"Writing test prediction to \033[1m{outfile}\033[0m")
    write_prediction_file(outfile, prediction)
    print(compute_score(prediction, test_data, True, "TEST ")[1])


def main(config: mmsrl.utils.dotdict) -> None:
    val_data = read_data("val")
    test_data = read_data("test")
    models_path = build_model_paths(config.prediction)
    models = read_predictions(models_path)
    if config.cut_off==0:
        for cut in range(31, 45, 2):
            print(cut/100)
            config.cut_off = cut /100
            perform_ensembling(config, models, val_data, test_data)
    else:
        perform_ensembling(config, models, val_data, test_data)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path, help="Path were we write the test prediction output.")
    parser.add_argument("prediction", type=pathlib.Path, nargs='+', help="Paths to directories containing prediction files, the filenames should contain \"val\" and \"test\" such that substituting one for other gives the corresponding predictions on the other split.")
    parser.add_argument("--aggregator", type=str, default="mean", help="How to aggregate results.")
    parser.add_argument("--full-matrix", action="store_true", help="When using logistic regression, learn a full 4*nmodel→4 matrix instead of a 4→1 one.")
    parser.add_argument("--cut-off", type=float, default=0.3, help="Ignore models with a lower F1 on the validation set.")
    parser.add_argument("--show-all", action="store_true", help="Show validation score of all (not cut-off) models.")
    parser.add_argument("--save-xgboost", action="store_true", help="Save the xgboost model and mapping.")
    parser.add_argument("--xgboost-params", type=str, default="{}", help="Should be eval()uable to a dict which is then added to xgboost parameters.")
    args = mmsrl.utils.dotdict(vars(parser.parse_args()))
    main(args)
