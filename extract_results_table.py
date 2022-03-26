from typing import Any, Dict, List, Optional, Tuple
import argparse
import collections
import functools
import math
import os
import pathlib
import re

import numpy
import scipy.stats
import statsmodels.stats.contingency_tables

import mmsrl.ensembling
import mmsrl.utils

RESULT_REGEX = re.compile(r"^Epoch\s+(\d+) (val|test), Macro-F1: (\d\.\d{3}), hero\[F(\d\.\d{2})/P(\d\.\d{2})/R(\d\.\d{2})\], villain\[F(\d\.\d{2})/P(\d\.\d{2})/R(\d\.\d{2})\], victim\[F(\d\.\d{2})/P(\d\.\d{2})/R(\d\.\d{2})\], other\[F(\d\.\d{2})/P(\d\.\d{2})/R(\d\.\d{2})\]$")
RESULT_KEYS = ["epoch", "split", "macro-F1", "hero F1", "hero P", "hero R", "villain F1", "villain P", "villain R", "victim F1", "victim P", "victim R", "other F1", "other P", "other R"]
CONFIG_REGEX = re.compile(r"^([^:]*): (.*)$")
CONFIG_UNIQUE_KEYS = ["output_test", "output_val"]


def all_equals(l: List[str]) -> bool:
    return all(x == l[0] for x in l[1:])


def config_to_run(config: Dict[str, str]) -> Any:
    key = config.copy()
    for unique in CONFIG_UNIQUE_KEYS:
        key.pop(unique, None)
    return tuple(sorted(key.items()))


def run_to_config(run: Any) -> Dict[str, str]:
    return dict(list(run))


@functools.cache
def levenshtein(lhs: str, rhs: str) -> int:
    if not lhs or not rhs:
        return len(lhs) + len(rhs)
    if lhs[0] == rhs[0]:
        return levenshtein(lhs[1:], rhs[1:])
    return min(2+levenshtein(lhs[1:], rhs), 2+levenshtein(lhs, rhs[1:]), 1+levenshtein(lhs[1:], rhs[1:]))


def get_experiments(args: argparse.Namespace, path: Optional[pathlib.Path] = None) -> Dict[str, List[pathlib.Path]]:
    if path is None:
        path = args.path
    experiments: Dict[str, List[pathlib.Path]] = {}
    sub_experiments: Dict[str, List[pathlib.Path]] = {}
    for filename in sorted(os.listdir(path)):
        if (path / filename).is_dir():
            if filename not in args.exclude:
                sub_experiments.update(get_experiments(args, path / filename))
            continue
        if not (filename.startswith(args.prefix) and filename.endswith(args.suffix)):
            continue

        matched: bool = False
        for candidate, filenames in experiments.items():
            if levenshtein(filename, candidate) < args.group_threshold:
                filenames.append(path / filename)
                matched = True
                break

        if not matched:
            experiments[filename] = [path / filename]

    return experiments | sub_experiments


def get_result(args: argparse.Namespace, data: List[str]) -> Dict[str, float]:
    last = {}
    for line in data:
        match = RESULT_REGEX.fullmatch(line)
        if match:
            last[match[2]] = match

    if args.split not in last or int(last[args.split][1]) < args.min_epoch:
        return {}

    result = dict(zip(RESULT_KEYS, last[args.split].groups()))
    result.pop("split")
    return {key: float(value) for key, value in result.items()}


def get_config(data: List[str]) -> Dict[str, str]:
    if not data or data[0] != "\033[33mConfiguration:\033[0m":
        return {}

    config: Dict[str, str] = {}
    for line in data[1:]:
        if "\033" in line or line.startswith("Epoch "):
            break

        match = CONFIG_REGEX.fullmatch(line)
        if match:
            config[match[1]] = match[2]
        else:
            break
    return config


def merge_configs(configs: Dict[pathlib.Path, Dict[str, str]]) -> Tuple[Dict[str, str], Dict[pathlib.Path, Dict[str, str]]]:
    transpose: Dict[str, List[str]] = collections.defaultdict(list)
    for config in configs.values():
        for key, value in config.items():
            transpose[key].append(value)

    base_config: Dict[str, str] = {}
    for key, values in transpose.items():
        if all_equals(values) and len(values) == len(configs):
            base_config[key] = values[0]

    delta_configs: Dict[pathlib.Path, Dict[str, str]] = {}
    for path, config in configs.items():
        delta_configs[path] = {key: value for key, value in config.items() if key not in base_config}
        
    return base_config, delta_configs


def merge_runs(configs: Dict[pathlib.Path, Dict[str, str]]) -> Dict[Any, List[pathlib.Path]]:
    runs: Dict[List[Tuple[str, str]], List[pathlib.Path]] = collections.defaultdict(list)
    for path, config in configs.items():
        runs[config_to_run(config)].append(path)
    return runs


def aggregate_results(results: List[Dict[str, float]]):
    aggregated: Dict[str, Dict[str, Any]] = {}
    for key in results[0].keys():
        values = numpy.array([result[key] for result in results])
        aggregated[key] = {
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
            "values": values
        }
    return aggregated


def merge_results(runs: Dict[Any, List[pathlib.Path]], results: Dict[pathlib.Path, Dict[str, float]]) -> Dict[Any, Dict[str, Any]]:
    merged: Dict[Any, Dict[str, Any]] = {}
    for run, paths in runs.items():
        merged[run] = {}
        merged[run]["runs"] = [results[path] for path in paths],
        merged[run].update(aggregate_results([results[path] for path in paths]))
    return merged


def mark_best(results: Dict[Any, Dict[str, Any]]) -> None:
    best = collections.defaultdict(lambda: 0)
    for run in results.values():
        for key, values in run.items():
            if isinstance(values, dict) and isinstance(values.get("mean"), float):
                best[key] = max(best[key], values["mean"])

    for run in results.values():
        for key, values in run.items():
            if isinstance(values, dict) and isinstance(values.get("mean"), float):
                values["best"] = (values["mean"] >= best[key])


def get_predictions(args: argparse.Namespace, runs: Dict[Any, List[pathlib.Path]], delta_configs: Dict[pathlib.Path, Dict[str, str]]) -> Tuple[Dict[Any, numpy.ndarray], numpy.ndarray]:
    predictions: Dict[Tuple[str, str], Dict[Any, List[int]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    gold = {sample["image"]: sample for sample in mmsrl.ensembling.read_data(args.split)}

    for run, paths in runs.items():
        for stdout_path in paths:
            base_path = stdout_path.parent
            prediction_path = pathlib.Path(delta_configs[stdout_path][f"output_{args.split}"])
            prediction = mmsrl.ensembling.read_prediction_file(base_path / prediction_path.parts[-1])
            for image, entities in prediction.items():
                for entity, probabilities in entities.items():
                    predictions[(image, entity)][run].append(probabilities.argmax())

    dataset_size: int = len(predictions)
    num_seeds: int = len(paths)
    prediction_array = {run: numpy.empty((dataset_size, num_seeds), dtype=numpy.int32) for run in runs}
    gold_array = numpy.empty((dataset_size,), dtype=numpy.int32)

    for i, ((image, entity), run_predictions) in enumerate(predictions.items()):
        assigned = False
        for ilabel, label in enumerate(mmsrl.utils.LABELS):
            if entity in gold[image][label]:
                gold_array[i] = ilabel
                assigned = True
        assert(assigned)

        for run, seed_predictions in run_predictions.items():
            prediction_array[run][i] = seed_predictions
    return prediction_array, gold_array


def compute_contingency_matrix(aggregate: str, lhs: numpy.ndarray, rhs: numpy.ndarray) -> numpy.ndarray:
    contingency = numpy.zeros((4, 4), dtype=numpy.int32)
    if aggregate == "first":
        for y, x in zip(lhs[:, 0], rhs[:, 0]):
            contingency[y, x] += 1
    elif aggregate == "flatten":
        for y, x in zip(lhs.flatten(), rhs.flatten()):
            contingency[y, x] += 1
    elif aggregate == "cartesian mean":
        for ys, xs in zip(lhs, rhs):
            for y in ys:
                for x in xs:
                    contingency[y, x] += 1
        contingency = contingency // (len(ys) * len(xs))
    return contingency


def compute_contingency_matrices(aggregate: str, predictions: Dict[Any, numpy.ndarray]) -> Dict[Tuple[Any, Any], numpy.ndarray]:
    contingencies = {}
    for run_lhs, lhs in predictions.items():
        for run_rhs, rhs in predictions.items():
            contingencies[(run_lhs, run_rhs)] = compute_contingency_matrix(aggregate, lhs, rhs)
    return contingencies


def compute_mcnemar_bowker(contingency: numpy.ndarray) -> float:
    return statsmodels.stats.contingency_tables.mcnemar(contingency, exact=False, correction=True).pvalue


def compute_macro_f1_t_test(lhs: Dict[str, Any], rhs: Dict[str, Any], equal_var: bool) -> float:
    return scipy.stats.ttest_ind(lhs["macro-F1"]["values"], rhs["macro-F1"]["values"], equal_var=equal_var, alternative="greater").pvalue


def compute_accuracies_t_test(lhs: numpy.ndarray, rhs: numpy.ndarray, gold: numpy.ndarray, equal_var: bool) -> float:
    lhs = (lhs == gold[:, None]).flatten()
    rhs = (rhs == gold[:, None]).flatten()
    return scipy.stats.ttest_ind(lhs, rhs, equal_var=equal_var, alternative="greater").pvalue


def compute_pvalues(args: argparse.Namespace, runs: Dict[Any, List[pathlib.Path]], delta_configs: Dict[pathlib.Path, Dict[str, str]], run_results: Dict[Any, Dict[str, Any]]) -> Dict[str, Dict[Tuple[Any, Any], float]]:
    if args.all_test:
        predictions, gold = get_predictions(args, runs, delta_configs)
        contingencies: Dict[str, Dict[Tuple[Any, Any], numpy.ndarray]] = {
                aggregation: compute_contingency_matrices(aggregation, predictions)
                for aggregation in ["first", "flatten", "cartesian mean"]
            }

    pvalues: Dict[str, Dict[Tuple[Any, Any], float]] = collections.defaultdict(dict)
    for run_lhs in runs:
        for run_rhs in runs:
            nseed = len(runs[run_lhs])
            pvalues[f'one-sided t-test of macro-F1'][(run_lhs, run_rhs)] = compute_macro_f1_t_test(run_results[run_lhs], run_results[run_rhs], equal_var=True)
            pvalues[f'one-sided Welch\'s unequal variances t-test of macro-F1'][(run_lhs, run_rhs)] = compute_macro_f1_t_test(run_results[run_lhs], run_results[run_rhs], equal_var=False)

            if args.all_test:
                pvalues[f'one-sided t-test of accuracies ({nseed} seed{"s" if nseed>1 else ""})'][(run_lhs, run_rhs)] = compute_accuracies_t_test(predictions[run_lhs], predictions[run_rhs], gold, equal_var=True)
                pvalues[f'one-sided Welch\'s unequal variances t-test of accuracies ({nseed} seed{"s" if nseed>1 else ""})'][(run_lhs, run_rhs)] = compute_accuracies_t_test(predictions[run_lhs], predictions[run_rhs], gold, equal_var=False)

                for aggregation, contingency in contingencies.items():
                    pvalues[f"McNemar-Bowker ({aggregation})"][(run_lhs, run_rhs)] = compute_mcnemar_bowker(contingency[(run_lhs, run_rhs)])
    return pvalues


def pre_html() -> None:
    print("""<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
<head>
    <title>MMSRL results</title>
    <meta charset="UTF-8" />
    <style type="text/css">
        /*<![CDATA[*/
        .pvalue {
            paint-order: stroke fill;
            -webkit-text-stroke: 10px rgba(255, 255, 255, .3);
        }
        table {
            border-collapse: collapse;
        }
        table, td, th {
            border: 1px solid;
        }
        td, th {
            max-width: 600px;
            padding: 4px;
        }
        dl.base_config > dt:before {
            content: "";
            display: block;
        }
        .best_true {
            font-weight: bold;
            color: #800;
        }
        dt { font-weight: bold; }
        dt:after { content: "="; }
        dt, dd {
            display: inline;
            margin: 0;
        }
        /*]]>*/
    </style>
</head>
<body>""")


def output_html_config(config: Dict[str, str], style: str) -> None:
    print(f'<dl class="{style}">')
    for key, value in config.items():
        print(f'<dt>{key}</dt><dd>{value}</dd>')
    print('</dl>')


def output_html_pvalue_color(pvalue: float, better: bool) -> None:
    if not numpy.isfinite(pvalue):
        hue = 200
    else:
        hue: int = round(min(-math.log(max(pvalue, 1e-10), 10), 2.5)/2.5*120)
    saturation: int = 100 if better else 25
    print(f'<td class="pvalue" style="background-color: hsl({hue}, {saturation}%, 50%);">{pvalue:.2e}</td>', end='')


def output_html(experiment: str, base_config: Dict[str, str], runs: Dict[Any, List[pathlib.Path]], run_results: Dict[Any, Dict[str, Any]], pvalues: Dict[str, Dict[Tuple[Any, Any], float]]) -> None:
    run_order = sorted(list(run_results.keys()))

    print('<section>')
    print(f'<h1>{experiment}</h1>')
    print('<details>')
    print('<summary>Base config</summary>')
    output_html_config(base_config, "base_config")
    print('</details>')
    print('<table>')
    print('    <thead>')
    print('        <tr><th rowspan="2">ID</th><th rowspan="2">Config</th><th rowspan="2">Macro-F1</th><th colspan="3">hero</th><th colspan="3">villain</th><th colspan="3">victim</th><th colspan="3">other</th></tr>')
    print('        <tr><th>F1</th><th>P</th><th>R</th><th>F1</th><th>P</th><th>R</th><th>F1</th><th>P</th><th>R</th><th>F1</th><th>P</th><th>R</th></tr>')
    print('    </thead>')
    print('    <tbody>')
    for irun, run in enumerate(run_order):
        result = run_results[run]
        print('        <tr>')
        print(f'            <td>{chr(ord("α")+irun)}</td>')
        print('            <td>', end='')
        print('<details><summary><code>', end='')
        config = run_to_config(run)
        output_html_config(config, "inline_config")
        print('</code></summary>')
        print('                <ul>')
        for path in runs[run]:
            print(f'                    <li>{path}</li>')
        print('                </ul></details>')
        print('</td>')
        for key in ["macro-F1", "hero F1", "hero P", "hero R", "villain F1", "villain P", "villain R", "victim F1", "victim P", "victim R", "other F1", "other P", "other R"]:
            data = result[key]
            details = "\n".join(f"{key}: {value}" for key, value in data.items())
            print('            <td>', end='')
            print(f'<abbr title="{details}" class="best_{data["best"]}">{data["mean"]:.2f}</abbr>', end='')
            print('</td>')
        print('        </tr>')
    print('    </tbody>')
    print('</table>')

    for test, pvalue in pvalues.items():
        print('<section>')
        print(f'<details><summary><h3>{test}</h3></summary>')
        print('<table>')
        print('    <tr>')
        print('        <th></th>')
        for irun in range(len(run_order)):
            print(f'        <th>{chr(ord("α")+irun)}</th>')
        print('    </tr>')
        for irun, lhs in enumerate(run_order):
            print('    <tr>')
            print(f'        <th>{chr(ord("α")+irun)}</th>')
            for rhs in run_order:
                output_html_pvalue_color(pvalue[(lhs, rhs)], run_results[lhs]["macro-F1"]["mean"] > run_results[rhs]["macro-F1"]["mean"])
            print('    </tr>')
        print('</table></details>')
        print('</section>')

    print('</section>')


def post_html() -> None:
    print('</body>')
    print('</html>')


def pre_latex() -> None:
    print(r"""\documentclass[a4paper, landscape]{article}
\usepackage[a4paper,margin=1in,landscape]{geometry}
\usepackage{fontspec}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{multirow}
\begin{document}""")


def output_latex_escape(x: str):
    return x.replace('_', r'\_').replace('[', r'\([\)').replace(']', r'\(]\)').replace('{', r'\({\)').replace('}', r'\(}\)')


def output_latex(experiment: str, base_config: Dict[str, str], runs: Dict[Any, List[pathlib.Path]], run_results: Dict[Any, Dict[str, Any]], pvalues: Dict[str, Dict[Tuple[Any, Any], float]]) -> None:
    print(f'\\section{{{output_latex_escape(experiment)}}}')
    print('\paragraph{Base config:}')
    print(r'\begin{description}[nosep]')
    for key, value in base_config.items():
        key = output_latex_escape(key)
        print(f'\\item[{key}=] \\verb|{value}|')
    print(r'\end{description}')
    print(r'\begin{tabular}{p{5cm} r r r r r r r r r r r r r}')
    print(r'    \toprule')
    print(r'    \multirow{2}{*}{Config} & \multirow{2}{*}{Macro-\(F_1\)} & \multicolumn{3}{c}{hero} & \multicolumn{3}{c}{villain} & \multicolumn{3}{c}{victim} & \multicolumn{3}{c}{other} \\')
    print(r'    & & \(F_1\) & \(P\) & \(R\) & \(F_1\) & \(P\) & \(R\) & \(F_1\) & \(P\) & \(R\) & \(F_1\) & \(P\) & \(R\) \\')
    print(r'    \midrule')
    for run, result in run_results.items():
        config = run_to_config(run)
        for key, value in config.items():
            key = output_latex_escape(key)
            print(f'    \\strong{{{key}=}} \\verb|{value}|', end='')
        for key in ["macro-F1", "hero F1", "hero P", "hero R", "villain F1", "villain P", "villain R", "victim F1", "victim P", "victim R", "other F1", "other P", "other R"]:
            data = result[key]
            if data["best"]:
                print(f' & \strong{{{data["mean"]:.2f}}}', end='')
            else:
                print(f' & {data["mean"]:.2f}', end='')
        print(r'\\')
    print(r'    \bottomrule')
    print(r'\end{tabular}')


def post_latex() -> None:
    print(r"\end{document}")


def main(args: argparse.Namespace) -> None:
    if args.html:
        pre, output, post = pre_html, output_html, post_html
    elif args.latex:
        pre, output, post = pre_latex, output_latex, post_latex
    pre()
    experiments = get_experiments(args)
    for experiment, paths in experiments.items():
        configs = {}
        results = {}
        for path in paths:
            with path.open("r") as file:
                data = list(map(lambda line: line.rstrip("\n"), file.readlines()))
            result = get_result(args, data)
            config = get_config(data)
            if result and config:
                results[path] = result
                configs[path] = config
        if not configs:
            continue
        base_config, delta_configs = merge_configs(configs)
        runs = merge_runs(delta_configs)
        run_results = merge_results(runs, results)
        mark_best(run_results)
        pvalues = compute_pvalues(args, runs, delta_configs, run_results)
        output(experiment, base_config, runs, run_results, pvalues)
    post()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-test", action="store_true", help="Compute all significance test methods.")
    parser.add_argument("--exclude", type=str, nargs='+', default=["old"], help="Ignore these directories")
    parser.add_argument("--group-threshold", type=int, default=10, help="Maximum levenshtein (add=del=2, mod=1) between experiences grouped in the same experiment.")
    parser.add_argument("--html", action="store_true", help="Output detailed table in HTML format.")
    parser.add_argument("--latex", action="store_true", help="Output table in LaTeX format.")
    parser.add_argument("--min-epoch", type=int, default=25, help="Ignore runs that did not reached this epoch number.")
    parser.add_argument("--prefix", type=str, default="", help="Prefix of the stdout files to consider, can be used to select a specific experiment.")
    parser.add_argument("--split", type=str, default="test", help="Output results on the given split (val or test).")
    parser.add_argument("--suffix", type=str, default=".out", help="Suffix of the stdout files to consider.")
    parser.add_argument("path", type=pathlib.Path, help="Path to the directory containing stdout files.")
    args = parser.parse_args()
    if args.html == args.latex:
        raise RuntimeError("You must use exactly one of --html or --latex.")
    main(args)
