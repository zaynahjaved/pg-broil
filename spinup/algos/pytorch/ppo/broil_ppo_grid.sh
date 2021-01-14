#!/bin/sh

for lambda in 0 0.1 0.2 0.8 0.9 1
do
  for alpha in 0.95
  do
      python broil_ppo.py --env PointBot-v0 --broil_lambda $lambda  --broil_alpha $alpha
  done
done
