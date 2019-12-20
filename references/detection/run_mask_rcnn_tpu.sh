#!/bin/bash
# Prerequisite: conda activate torch-xla-nightly-vision
set -e

# TPU XLA
export TPU_IP_ADDRESS="10.2.101.2"
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

# CPU XLA
#export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
#export XRT_WORKERS="localservice:0;grpc://localhost:40934"

# Helps us but still buggy, still have to use padding a lot
export XLA_EXPERIMENTAL=nonzero:masked_select

python ~/vision/references/detection/train_tpu.py "${@}"
