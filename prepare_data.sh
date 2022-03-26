#!/bin/sh

if [ -z "$DATA_PATH" ]; then
	echo "Error: Variable DATA_PATH is not set."
	echo "Please set it to the path where you want to store the data."
	echo "For example, from the root of the git repository you can type:"
	echo '    export DATA_PATH="$PWD/../data"'
	exit 1
fi

if [ ! -f "$DATA_PATH/mmsrl/constraint-data.zip" ]; then
	echo 'Data not found at $DATA_PATH/mmsrl/constraint-data.zip.'
	echo "For information DATA_PATH is set to: $(readlink -m $DATA_PATH)"
	echo "Downloading dataset..."
	mkdir -p "$DATA_PATH/mmsrl"
	curl "https://esimon.eu/tmp/constraint-data.zip" --output "$DATA_PATH/mmsrl/constraint-data.zip"
fi

if [ ! -f "$DATA_PATH/mmsrl/constraint22_dataset_unseen_test.zip" ]; then
	echo 'Test data not found at $DATA_PATH/mmsrl/constraint22_dataset_unseen_test.zip.'
	echo "For information DATA_PATH is set to: $(readlink -m $DATA_PATH)"
	echo "Downloading test dataset..."
	curl "https://esimon.eu/tmp/constraint22_dataset_unseen_test.zip" --output "$DATA_PATH/mmsrl/constraint22_dataset_unseen_test.zip"
fi

echo "Preparing data in $DATA_PATH"
echo "Unzipping"
unzip -q "$DATA_PATH/mmsrl/constraint-data.zip" -d "$DATA_PATH/mmsrl"
unzip -q "$DATA_PATH/mmsrl/constraint22_dataset_covid19.zip" -d "$DATA_PATH/mmsrl/covid19"
unzip -q "$DATA_PATH/mmsrl/constraint22_dataset_uspolitics.zip" -d "$DATA_PATH/mmsrl/uspolitics"
unzip -q "$DATA_PATH/mmsrl/constraint22_dataset_unseen_test.zip" -d "$DATA_PATH/mmsrl/all"

if [ ! -f "$DATA_PATH/mmsrl/all/annotations/test.jsonl" ]; then
	echo 'Labeled test data not found at $DATA_PATH/mmsrl/all/annotations/test.jsonl.'
	echo "For information DATA_PATH is set to: $(readlink -m $DATA_PATH)"
	echo "Downloading test dataset..."
	curl "https://esimon.eu/tmp/constraint22_dataset_unseen_test_gold_labels.jsonl" --output "$DATA_PATH/mmsrl/all/annotations/test.jsonl"
fi

echo "Merging datasets"
cat "$DATA_PATH/mmsrl/covid19/annotations/train.jsonl" "$DATA_PATH/mmsrl/uspolitics/annotations/train.jsonl" | grep -v '"hero": \[\], "villain": \[\], "victim": \[\], "other": \[\]' > "$DATA_PATH/mmsrl/all/annotations/train.jsonl"
cat "$DATA_PATH/mmsrl/covid19/annotations/val.jsonl" "$DATA_PATH/mmsrl/uspolitics/annotations/val.jsonl" | grep -v '"hero": \[\], "villain": \[\], "victim": \[\], "other": \[\]' > "$DATA_PATH/mmsrl/all/annotations/val.jsonl"
cp -rl "$DATA_PATH/mmsrl/covid19/images" "$DATA_PATH/mmsrl/all"
cp -rl "$DATA_PATH/mmsrl/uspolitics/images" "$DATA_PATH/mmsrl/all"
rm -r "$DATA_PATH/mmsrl/covid19/images"
rm -r "$DATA_PATH/mmsrl/uspolitics/images"
ln -s "$DATA_PATH/mmsrl/all/images" "$DATA_PATH/mmsrl/covid19/images"
ln -s "$DATA_PATH/mmsrl/all/images" "$DATA_PATH/mmsrl/uspolitics/images"
