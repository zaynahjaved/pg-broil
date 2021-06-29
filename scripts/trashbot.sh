python ../spinup/algos/pytorch/rex/brex/brex_basic.py --features_dir ../demonstrations/trashbot_demos --save_file ../spinup/rewards/brex_reward_trashbot.pkl --env PointBot-v0 --normalize
python ../spinup/algos/pytorch/ppo/broil_ppo.py --env PointBot-v0 --brex True
