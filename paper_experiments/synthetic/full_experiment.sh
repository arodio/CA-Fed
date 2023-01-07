cd ../../

DEVICE="cpu"
SEED=42

N_ROUNDS=150
LOCAL_STEPS=2
LOG_FREQ=10

echo "Experiment with synthetic clustered dataset"

echo "=> Generate data.."

cd data/ || exit

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

cd ../

echo "=> Run experiment with 'unbiased' clients sampler"

name="unbiased"

python run_experiment.py \
  --experiment synthetic \
  --cfg_file_path data/synthetic/cfg.json \
  --objective_type weighted \
  --aggregator_type centralized \
  --clients_sampler "${name}" \
  --n_rounds "${N_ROUNDS}" \
  --local_steps "${LOCAL_STEPS}" \
  --local_optimizer sgd \
  --local_lr 0.03 \
  --server_optimizer sgd \
  --server_lr 0.03 \
  --train_bz 32 \
  --test_bz 1024 \
  --device "${DEVICE}" \
  --log_freq "${LOG_FREQ}" \
  --verbose 1 \
  --logs_dir "logs/synthetic/activity_${name}/seed_${SEED}" \
  --history_path "history/synthetic/activity_${name}/seed_${SEED}.json" \
  --seed "${SEED}"

echo "=> Run experiment with 'markov' clients sampler"

name="markov"

python run_experiment.py \
  --experiment synthetic \
  --cfg_file_path data/synthetic/cfg.json \
  --objective_type weighted \
  --aggregator_type centralized \
  --clients_sampler "${name}" \
  --smoothness_param 0.2 \
  --tolerance_param 0.0 \
  --n_rounds "${N_ROUNDS}" \
  --local_steps "${LOCAL_STEPS}" \
  --local_optimizer sgd \
  --local_lr 0.01 \
  --server_optimizer sgd \
  --server_lr 0.3 \
  --train_bz 32 \
  --test_bz 1024 \
  --device "${DEVICE}" \
  --log_freq "${LOG_FREQ}" \
  --verbose 1 \
  --logs_dir "logs/synthetic/activity_${name}/seed_${SEED}" \
  --history_path "history/synthetic/activity_${name}/seed_${SEED}.json" \
  --seed "${SEED}"

echo "=> Run experiment with 'adafed' clients sampler"

name="adafed"

python run_experiment.py \
  --experiment synthetic \
  --cfg_file_path data/synthetic/cfg.json \
  --objective_type weighted \
  --aggregator_type centralized \
  --clients_sampler "${name}" \
  --adafed_full_participation \
  --n_rounds "${N_ROUNDS}" \
  --local_steps "${LOCAL_STEPS}" \
  --local_optimizer sgd \
  --local_lr 0.001 \
  --server_optimizer sgd \
  --server_lr 1.0 \
  --train_bz 32 \
  --test_bz 1024 \
  --device "${DEVICE}" \
  --log_freq "${LOG_FREQ}" \
  --verbose 1 \
  --logs_dir "logs/synthetic/activity_${name}/seed_${SEED}" \
  --history_path "history/synthetic/activity_${name}/seed_${SEED}.json" \
  --seed "${SEED}"

echo "=> Run experiment with 'fast' clients sampler"

name="fast"

python run_experiment.py \
  --experiment synthetic \
  --cfg_file_path data/synthetic/cfg.json \
  --objective_type weighted \
  --aggregator_type centralized \
  --clients_sampler "${name}" \
  --smoothness_param 0.001 \
  --fast_n_clients_per_round 12 \
  --n_rounds "${N_ROUNDS}" \
  --local_steps "${LOCAL_STEPS}" \
  --local_optimizer sgd \
  --local_lr 0.03 \
  --server_optimizer sgd \
  --server_lr 0.03 \
  --train_bz 32 \
  --test_bz 1024 \
  --device "${DEVICE}" \
  --log_freq "${LOG_FREQ}" \
  --verbose 1 \
  --logs_dir "logs/synthetic/activity_${name}/seed_${SEED}" \
  --history_path "history/synthetic/activity_${name}/seed_${SEED}.json" \
  --seed "${SEED}"

echo "=> Generate plots.."

python make_plots.py \
  --logs_dir "logs/synthetic/" \
  --history_dir "history/synthetic/" \
  --save_dir "plots/synthetic"
