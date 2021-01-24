#!/bin/sh

for lambda in 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59
do
  for alpha in 0.95
  do
      python broil_ppo.py --env PointBot-v0 --broil_lambda $lambda  --broil_alpha $alpha
  done
done
