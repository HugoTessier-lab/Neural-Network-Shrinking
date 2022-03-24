#Arguments: $1: path to the model to benchmark, $2: file in which to store tegrastats' output, $3: pruning rate, $4: input shape, $5: duration of sleep in seconds
sudo tegrastats --logfile $2 &
sleep $5s
OPENBLAS_CORETYPE=CORTEXA57 python3 benchmark_on_jetson.py --model $1 --pruning_rate $3 --input_shape $4
sleep $5s
sudo tegrastats --stop
sudo pkill tegrastats