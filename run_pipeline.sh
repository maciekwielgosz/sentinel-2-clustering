#!/bin/bash

python3 prepare_data.py
python3 compute_svd.py
python3 cluster.py