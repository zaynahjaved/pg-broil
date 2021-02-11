#!/bin/sh

for lambda in 0 0.2 0.4 0.45 0.6 0.8 1.0
do
  for alpha in 0.96
  do
      python broil_ppo.py --env PointBot-v0 --broil_lambda $lambda  --broil_alpha $alpha --risk_metric 'erm'
  done
done
