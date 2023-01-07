DEVICE=cpu

cd ../../data/ || exit

rm -r synthetic

python main.py \
  --dataset synthetic_clustered \
  --n_tasks 24 \
  --n_clusters 2 \
  --dimension 10 \
  --n_train_samples 50 \
  --n_test_samples 150 \
  --hetero_param 0.2 \
  --availability_parameters 0.4 0.4 \
  --stability_parameters 0.9 0.9 \
  --deterministic_split \
  --save_dir synthetic \
  --seed 42

cd ../paper_experiments/synthetic || exit

chmod +x run.sh

for seed in 12 123 1234 12345 57 1453 1927 1956 2011
do
  ./run.sh -n "markov" -s "${seed}" -d "${DEVICE}" -l 0.01 -r 0.3 -b 1.0
done


