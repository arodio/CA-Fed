helpFunction()
{
   echo ""
   echo "Usage: $0 -n name -s seed -d device -l local_lr -b bias"
   echo -e "\t-n name of the experiment"
   echo -e "\t-s seed to be used for experiment"
   echo -e "\t-d device to be used, e.g., cuda or cpu"
   echo -e "\t-l client learning rate"
   echo -e "\t-r server learning rate"
   echo -e "\t-b bias multiplicative coefficient"
   exit 1 # Exit script after printing help
}

while getopts "n:s:d:l:r:b:" opt
do
   case "$opt" in
      n ) name="$OPTARG" ;;
      s ) seed="$OPTARG" ;;
      d ) device="$OPTARG" ;;
      l ) local_lr="$OPTARG" ;;
      r ) server_lr="$OPTARG" ;;
      b ) bias="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done


# Print helpFunction in case parameters are empty
if [ -z "${name}" ] || [ -z "${seed}" ] || [ -z "${device}" ] || [ -z "${local_lr}" ] || [ -z "${server_lr}" ] || [ -z "${bias}" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi


cd ../../

N_ROUNDS=150
LOCAL_STEPS=2
LOG_FREQ=10

echo "=> name=${name} | seed=${seed} | local_lr=${local_lr} | server_lr=${server_lr} | bias_const=${bias}"

echo "==> Run experiment with '${name}' clients sampler"

python run_experiment.py \
  --experiment synthetic_clustered \
  --cfg_file_path data/synthetic/cfg.json \
  --objective_type weighted \
  --aggregator_type centralized \
  --clients_sampler "${name}" \
  --smoothness_param 0.001 \
  --fast_n_clients_per_round 12 \
  --adafed_full_participation \
  --tolerance_param 0.0 \
  --bias_const "${bias}" \
  --n_rounds "${N_ROUNDS}" \
  --local_steps "${LOCAL_STEPS}" \
  --local_optimizer sgd \
  --local_lr "${local_lr}" \
  --server_optimizer sgd \
  --server_lr "${server_lr}" \
  --train_bz 32 \
  --test_bz 1024 \
  --device "${device}" \
  --log_freq "${LOG_FREQ}" \
  --verbose 1 \
  --logs_dir "logs/synthetic/activity_${name}/seed_${seed}" \
  --history_path "history/synthetic/activity_${name}/seed_${seed}.json" \
  --seed "${seed}"
