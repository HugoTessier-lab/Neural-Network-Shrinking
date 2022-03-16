#Arguments: $1: file in which to store tegrastats' output, $2: pruning rate, $3: input shape, $4: duration of sleep in seconds
sudo tegrastats --logfile $1 &
sleep $4s
OPENBLAS_CORETYPE=CORTEXA57 python3 test3.py --pruning_rate $2 --input_shape $3
sleep $4s
sudo tegrastats --stop
