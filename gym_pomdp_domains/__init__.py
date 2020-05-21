from gym.envs.registration import register

register(
    id='Hallway-v0',
    entry_point='gym_pomdp_domains.envs:Hallway',
)

register(
    id='Hallway_Simple-v0',
    entry_point='gym_pomdp_domains.envs:Hallway_Simple',
)

register(
    id='Hallway2-v0',
    entry_point='gym_pomdp_domains.envs:Hallway2',
)