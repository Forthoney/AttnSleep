#!/usr/bin/env bash
data=$1

start=0
end=$(jq '.num_folds' < config.json)
end=$((end-1))

for i in $(eval echo "{$start..$end}")
do
   python train.py --fold_id="$i" --data_dir "$data"
done
