from optuna.distributions import FloatDistribution, IntDistribution

search_spaces = {
    'source': {
        'ppo': {
            'learning_rate': FloatDistribution(high=5e-3, low=1e-4),
            'batch_size': IntDistribution(high=256, low=128, step=128),
            'gamma': FloatDistribution(high=0.99, low=0.8),
            'ent_coef': FloatDistribution(high=1e-2, low=0),
            'vf_coef': FloatDistribution(high=0.75, low=0.25, step=0.05),
            'gae_lambda': FloatDistribution(high=0.99, low=0.9, step=0.01),
            'clip_range': FloatDistribution(high=0.4, low=0.1, step=0.1),
            'clip_range_vf': FloatDistribution(high=0.4, low=0.1, step=0.1),
        }
    },
    'target': {
        'ppo': {
            'learning_rate': FloatDistribution(high=5e-3, low=1e-4),
            'batch_size': IntDistribution(high=256, low=128, step=128),
            'gamma': FloatDistribution(high=0.99, low=0.8),
            'ent_coef': FloatDistribution(high=1e-2, low=0),
            'vf_coef': FloatDistribution(high=0.75, low=0.25, step=0.05),
            'gae_lambda': FloatDistribution(high=0.99, low=0.9, step=0.01),
            'clip_range': FloatDistribution(high=0.4, low=0.1, step=0.1),
            'clip_range_vf': FloatDistribution(high=0.4, low=0.1, step=0.1),
        }
    }
}

search_spaces_udr = {
    'source': {
        'ppo': {
            'learning_rate': FloatDistribution(high=5e-3, low=1e-4),
            'batch_size': IntDistribution(high=256, low=128, step=128),
            'gamma': FloatDistribution(high=0.99, low=0.8),
            'ent_coef': FloatDistribution(high=1e-2, low=0),
            'vf_coef': FloatDistribution(high=0.75, low=0.25, step=0.05),
            'gae_lambda': FloatDistribution(high=0.99, low=0.9, step=0.01),
            'clip_range': FloatDistribution(high=0.4, low=0.1, step=0.1),
            'clip_range_vf': FloatDistribution(high=0.4, low=0.1, step=0.1),
            'eps_udr': FloatDistribution(high=1, low=0)
        }
    },
    'target': {
        'ppo': {
            'learning_rate': FloatDistribution(high=5e-3, low=1e-4),
            'batch_size': IntDistribution(high=256, low=128, step=128),
            'gamma': FloatDistribution(high=0.99, low=0.8),
            'ent_coef': FloatDistribution(high=1e-2, low=0),
            'vf_coef': FloatDistribution(high=0.75, low=0.25, step=0.05),
            'gae_lambda': FloatDistribution(high=0.99, low=0.9, step=0.01),
            'clip_range': FloatDistribution(high=0.4, low=0.1, step=0.1),
            'clip_range_vf': FloatDistribution(high=0.4, low=0.1, step=0.1),
            'eps_udr': FloatDistribution(high=1, low=0)
        }
    }
}
