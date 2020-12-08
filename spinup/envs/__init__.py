from gym.envs.registration import register
register(
    id='PointBot-v0',
    entry_point='spinup.envs.pointbot:PointBot')
register(
    id='Shelf-v0',
    entry_point='spinup.envs.shelf_env:ShelfEnv')
