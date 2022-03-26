from typing import Any, Dict, List, Tuple

import numpy
import sklearn.metrics
import statsmodels.stats.inter_rater

import annotations.annotator
import mmsrl.utils


def show_scores(annotator: str, data: Dict[str, Any], answers: List[Tuple[str, str, int]]):
    truth = []
    predictions = []
    assert(len(answers) == 100)
    for image, entity, answer in answers:
        sample = data[image]
        truth.append(sample["labels"][sample["entities_name"].index(entity)].item())
        predictions.append(answer)

    f1 = sklearn.metrics.f1_score(truth, predictions, average="macro")
    class_precision, class_recall, class_f1, _ = sklearn.metrics.precision_recall_fscore_support(truth, predictions)
    class_scores = ", ".join(f"{label}[F{class_f1[i]:4.2f}/P{class_precision[i]:4.2f}/R{class_recall[i]:4.2f}]" for i, label in enumerate(mmsrl.utils.LABELS))
    print(f"Annotator {annotator}, Macro-F1: {f1:5.3f}, {class_scores}")
    return f1


def get_annotations(annotator: str) -> List[Tuple[str, str, int]]:
    answers = []
    with open(f"annotations/annotations of {annotator}", "r") as file:
        for line in file:
            image, entity, answer = line.rstrip('\n').split('\t')
            answers.append((image, entity, int(answer)))
    return answers


def main(annotators: List[str]):
    data, _ = annotations.annotator.get_data()
    answers = {}
    x = numpy.empty((100, len(annotators)), dtype=numpy.int32)
    m = numpy.zeros((100, 4), dtype=numpy.int32)
    scores = []
    for i, annotator in enumerate(annotators):
        answers[annotator] = get_annotations(annotator)
        f1 = show_scores(annotator, data, answers[annotator])
        scores.append(f1)
        for j, (_, _, answer) in enumerate(answers[annotator]):
            x[j, i] = answer
            m[j, answer] += 1
    print(f"Average Macro-F1: {numpy.mean(scores)*100:.1f}")
    print(f"Std Macro-F1: {numpy.std(scores)*100:.1f}")
    y = numpy.array([data[image]["labels"][data[image]["entities_name"].index(entity)].item() for image, entity, _ in answers[annotator]], dtype=numpy.int32)
    print("")
    
    print(f"Fleiss kappa: {statsmodels.stats.inter_rater.fleiss_kappa(m):.2f}")
    kappas = []
    for i, annotatorA in enumerate(annotators):
        for jmi, annotatorB in enumerate(annotators[i+1:]):
            j = i + 1 + jmi
            kappa = sklearn.metrics.cohen_kappa_score(x[:, i], x[:, j])
            kappas.append(kappa)
            print(f"Cohen kappa between annotators {annotatorA} and {annotatorB}: {kappa:.2f}")
    avg_kappa = numpy.mean(kappas)
    print(f"Average Cohen kappa: {avg_kappa:.2f}")


    print("")
    print(f"Percent of samples where all annotators agree: {numpy.equal(x[:, [0]], x[:, 1:]).all(axis=1).sum()}%")
    for i, annotatorA in enumerate(annotators):
        for jmi, annotatorB in enumerate(annotators[i+1:]):
            j = i + 1 + jmi
            print(f"Percent agreement between {annotatorA} and {annotatorB}: {(x[:, i] == x[:, j]).sum()}%")


if __name__ == "__main__":
    main(["A", "É", "S", "D", "U"])
