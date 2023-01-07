cd ../../

DEVICE="cpu"
SEED=42

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

N_ROUNDS=150
LOCAL_STEPS=2
LOG_FREQ=10

echo "==> Run experiment with 'markov' clients sampler"
name="markov"

for lr in 0.1 0.03 0.01 0.003 0.001
do
  for server_lr in 1.0 0.3 0.1 0.03 0.01
  do
    echo "=> activity=${name} | lr=${lr} | server_lr=${server_lr} | seed=${SEED}"
    python run_experiment.py \
      --experiment "mnist" \
      --cfg_file_path data/mnist/cfg.json \
      --objective_type weighted \
      --aggregator_type centralized \
      --clients_sampler "${name}" \
      --smoothness_param 0.0 \
      --tolerance_param 0.0 \
      --bias_const 1.0 \
      --n_rounds "${N_ROUNDS}" \
      --local_steps "${LOCAL_STEPS}" \
      --local_optimizer sgd \
      --local_lr "${lr}" \
      --server_optimizer sgd \
      --server_lr "${server_lr}" \
      --train_bz 64 \
      --test_bz 2048 \
      --device "${DEVICE}" \
      --log_freq "${LOG_FREQ}" \
      --verbose 1 \
      --logs_dir "logs_tuning/mnist/activity_${name}/mnist_lr_${lr}_server_${server_lr}/seed_${SEED}" \
      --history_path "history_tuning/mnist/activity_${name}/mnist_lr_${lr}_server_${server_lr}/seed_${SEED}.json" \
      --seed "${SEED}"
  done
done


echo "==> Run experiment with 'adafed' clients sampler"
name="adafed"

for lr in 0.003 0.001 0.0003 0.0001
do
  for server_lr in 1.0
  do
    echo "=> activity=${name} | lr=${lr} | server_lr=${server_lr} | seed=${SEED}"
    python run_experiment.py \
      --experiment "mnist" \
      --cfg_file_path data/mnist/cfg.json \
      --objective_type weighted \
      --aggregator_type centralized \
      --clients_sampler "${name}" \
      --adafed_full_participation \
      --n_rounds "${N_ROUNDS}" \
      --local_steps "${LOCAL_STEPS}" \
      --local_optimizer sgd \
      --local_lr "${lr}" \
      --server_optimizer sgd \
      --server_lr "${server_lr}" \
      --train_bz 64 \
      --test_bz 2048 \
      --device "${DEVICE}" \
      --log_freq "${LOG_FREQ}" \
      --verbose 1 \
      --logs_dir "logs_tuning/mnist/activity_${name}/mnist_lr_${lr}_server_${server_lr}/seed_${SEED}" \
      --history_path "history_tuning/mnist/activity_${name}/mnist_lr_${lr}_server_${server_lr}/seed_${SEED}.json" \
      --seed ${SEED}
  done
done

echo "==> Run experiment with 'unbiased' clients sampler"
name="unbiased"

for lr in 0.1 0.03 0.01 0.003 0.001
do
  for server_lr in 1.0 0.3 0.1 0.03 0.01
  do
    echo "=> activity=${name} | lr=${lr} | server_lr=${server_lr} | seed=${SEED}"
    python run_experiment.py \
      --experiment "mnist" \
      --cfg_file_path data/mnist/cfg.json \
      --objective_type weighted \
      --aggregator_type centralized \
      --clients_sampler "${name}" \
      --n_rounds "${N_ROUNDS}" \
      --local_steps "${LOCAL_STEPS}" \
      --local_optimizer sgd \
      --local_lr "${lr}" \
      --server_optimizer sgd \
      --server_lr "${server_lr}" \
      --train_bz 64 \
      --test_bz 2048 \
      --device "${DEVICE}" \
      --log_freq "${LOG_FREQ}" \
      --verbose 1 \
      --logs_dir "logs_tuning/mnist/activity_${name}/mnist_lr_${lr}_server_${server_lr}/seed_${SEED}" \
      --history_path "history_tuning/mnist/activity_${name}/mnist_lr_${lr}_server_${server_lr}/seed_${SEED}.json" \
      --seed "${SEED}"
  done
done


echo "==> Run experiment with 'fast' clients sampler"
name="fast"

for lr in 0.1 0.03 0.01 0.003 0.001
do
  for server_lr in 1.0 0.3 0.1 0.03 0.01
  do
    echo "=> activity=${name} | lr=${lr} | server_lr=${server_lr} | seed=${SEED}"
    python run_experiment.py \
      --experiment "mnist" \
      --cfg_file_path data/mnist/cfg.json \
      --objective_type weighted \
      --aggregator_type centralized \
      --clients_sampler "${name}" \
      --smoothness_param 0.001 \
      --fast_n_clients_per_round 12 \
      --n_rounds "${N_ROUNDS}" \
      --local_steps "${LOCAL_STEPS}" \
      --local_optimizer sgd \
      --local_lr "${lr}" \
      --server_optimizer sgd \
      --server_lr "${server_lr}" \
      --train_bz 64 \
      --test_bz 2048 \
      --device "${DEVICE}" \
      --log_freq "${LOG_FREQ}" \
      --verbose 1 \
      --logs_dir "logs_tuning/mnist/activity_${name}/mnist_lr_${lr}_server_${server_lr}/seed_${SEED}" \
      --history_path "history_tuning/mnist/activity_${name}/mnist_lr_${lr}_server_${server_lr}/seed_${SEED}.json" \
      --seed "${SEED}"
  done
done
