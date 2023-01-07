cd ../../

DEVICE="cuda"
SEED=42

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
      --experiment "cifar10" \
      --cfg_file_path data/cifar10/cfg.json \
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
      --logs_dir "logs_tuning/cifar/activity_${name}/cifar_lr_${lr}_server_${server_lr}/seed_${SEED}" \
      --history_path "history_tuning/cifar/activity_${name}/cifar_lr_${lr}_server_${server_lr}/seed_${SEED}.json" \
      --seed "${SEED}"
  done
done


echo "==> Run experiment with 'adafed' clients sampler"
name="adafed"

for lr in 0.1 0.03 0.01 0.003 0.001
do
  for server_lr in 1.0 0.3 0.1 0.03 0.01
  do
    echo "=> activity=${name} | lr=${lr} | server_lr=${server_lr} | seed=${SEED}"
    python run_experiment.py \
      --experiment "cifar10" \
      --cfg_file_path data/cifar10/cfg.json \
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
      --logs_dir "logs_tuning/cifar/activity_${name}/cifar_lr_${lr}_server_${server_lr}/seed_${SEED}" \
      --history_path "history_tuning/cifar/activity_${name}/cifar_lr_${lr}_server_${server_lr}/seed_${SEED}.json" \
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
      --experiment "cifar10" \
      --cfg_file_path data/cifar10/cfg.json \
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
      --logs_dir "logs_tuning/cifar/activity_${name}/cifar_lr_${lr}_server_${server_lr}/seed_${SEED}" \
      --history_path "history_tuning/cifar/activity_${name}/cifar_lr_${lr}_server_${server_lr}/seed_${SEED}.json" \
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
      --experiment "cifar10" \
      --cfg_file_path data/cifar10/cfg.json \
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
      --logs_dir "logs_tuning/cifar/activity_${name}/cifar_lr_${lr}_server_${server_lr}/seed_${SEED}" \
      --history_path "history_tuning/cifar/activity_${name}/cifar_lr_${lr}_server_${server_lr}/seed_${SEED}.json" \
      --seed "${SEED}"
  done
done
