DEVICE=cpu

cd ../../data/ || exit

rm -r mnist

python main.py \
  --dataset mnist \
  --n_tasks 24 \
  --incongruent_split \
  --n_clusters 2 \
  --n_swapping_pairs 2 \
  --availability_parameters 0.4 0.4 \
  --stability_parameters 0.9 0.9 \
  --save_dir mnist \
  --seed 42

cd ../paper_experiments/mnist || exit

chmod +x run.sh

for seed in 12 123 1234 12345 57 1453 1927 1956 2011
do
  ./run.sh -s "${seed}" -d "${DEVICE}"
done