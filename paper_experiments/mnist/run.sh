helpFunction()
{
   echo ""
   echo "Usage: $0 -s seed -d device"
   echo -e "\t-s seed to be used for experiment"
   echo -e "\t-d device to be used, e.g., cuda or cpu"
   exit 1 # Exit script after printing help
}

while getopts "s:d:" opt
do
   case "$opt" in
      s ) seed="$OPTARG" ;;
      d ) device="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done


# Print helpFunction in case parameters are empty
if [ -z "${seed}" ] || [ -z "${device}" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi


cd ../../


N_ROUNDS=150
LOCAL_STEPS=2
LOG_FREQ=10

echo "=> seed=${seed}"

echo "==> Run experiment with 'markov' clients sampler"

name="markov"
python run_experiment.py \
  --experiment "mnist" \
  --cfg_file_path data/mnist/cfg.json \
  --objective_type weighted \
  --aggregator_type centralized \
  --clients_sampler "${name}" \
  --smoothness_param 0.0 \
  --tolerance_param 0.0 \
  --n_rounds "${N_ROUNDS}" \
  --local_steps "${LOCAL_STEPS}" \
  --local_optimizer sgd \
  --local_lr 0.001 \
  --server_optimizer sgd \
  --server_lr 0.1 \
  --train_bz 64 \
  --test_bz 2048 \
  --device "${device}" \
  --log_freq "${LOG_FREQ}" \
  --verbose 1 \
  --logs_dir "logs/mnist/activity_${name}/mnist_lr_${lr}_server_${server_lr}/seed_${seed}" \
  --history_path "history/mnist/activity_${name}/mnist_lr_${lr}_server_${server_lr}/seed_${seed}.json" \
  --seed "${seed}"

