import pathlib
import pickle
import sys


def show_predictions(path: pathlib.Path) -> None:
    with path.open("rb") as input_file:
        try:
            while True:
                image, entities = pickle.load(input_file)
                for entity, prediction in entities.items():
                    prediction_str = " ".join(map(str, prediction.tolist()))
                    print(f"{image}\t{entity}\t{prediction_str}")
                print("")
        except EOFError:
            pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} prediction_file", file=sys.stderr)
        sys.exit(1)
    show_predictions(pathlib.Path(sys.argv[1]))
