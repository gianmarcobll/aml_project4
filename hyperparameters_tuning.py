import shutil

from env.custom_hopper import *
from functools import partial
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from policy_gradient import *

from optuna_search_spaces import search_spaces, search_spaces_udr
import optuna
import os

N_EVALUATIONS = 5
N_TIMESTEPS = 100_000
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 100

def check_path(path):
    """
    Check whether a path exists or not. If not it creates the path.
    
    :param path: path we want to check
    :type path: str
    :return: none
    """
    if not os.path.exists(path):
        os.makedirs(path)

def create_env(meta, env=None):
    """
    We use this function to generate the domain spaces used for traininng. If it is 'grid'
    we try to develop a grid search on domain randomization.
    
    :param meta: dictionary of the model we want to train
    :type meta: dict
    :return: training environment
    """
    
    env = gym.make(meta['env'])

    print('State space:', env.observation_space.shape)
    print('Action space:', env.action_space)
    print('Dynamics parameters:', env.get_parameters()) 

    # if meta['obs'] == 'cnn':
    #     if env:
    #         env = gym.make(env)
    #     else:
    #         env = gym.make(meta['env'])
    #     env = PixelObservationWrapper(env)
    #     print('state observation shape', env.observation_space.shape)
    #     if meta['preprocess']:
    #         env = PreprocessWrapper(env)
    #         print('obervation shape after preprocessing:', env.observation_space.shape)
            
    #     if meta['gray_scale']:
    #         env = GrayScaleWrapper(env, smooth=meta['smooth'], preprocessed=meta['preprocess'], keep_dim=False)
    #         print('obervation shape after gray scaling:', env.observation_space.shape)
    #         if meta['resize']:
    #             env = ResizeWrapper(env, shape=meta['resize_shape'])
    #             print('obervation shape after resizing:', env.observation_space.shape)

    #     if meta['env']:
    #         env = FrameStack(env, num_stack=meta['n_frame_stacks'])
    #         print('obervation shape after stacking:', env.observation_space.shape)
    
    # if meta['domain_randomization']:
    #     env.set_distributions(meta['chosen_domain'])
        
    return env

def create_model_path(models_path):
    """
    create path for the current model.
    
    :param models_path: path to all of the models
    :type models_path: 'str'
    :return: model path
    """
    paths = []
    
    for file in os.listdir(models_path):
        if os.path.isdir(os.path.join(models_path, file)):
            paths.append(int(file.split('_')[-1]))
    
    if len(paths) == 0:
        model_num = 1
        new_path = os.path.join(models_path, f'model_{model_num}')
    else:
        
        model_num = max(paths)
        model_path = os.path.join(models_path, f'model_{model_num}')
        count = 0
        for file in os.listdir(model_path):
            if os.path.isfile(os.path.join(model_path, file)):
                count += 1

        if count == 1:
            shutil.rmtree(model_path)
            new_path = model_path
            
        else:
            new_path = os.path.join(models_path, f'model_{model_num+1}')
                

    return new_path

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
            self,
            eval_env: gym.Env,
            trial: optuna.Trial,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def main(params):
    '''
    Another approach for hyperparameter tuning is using optuna.
    this approach have been supported by the the stable_baselines3.
    
    In this method we can choose n number of trials, and the model
    will be trained for n set of hyper parameters (all hyperparameters
    chosen randomly with a uniform distribution). 
    
    In brute force approach optimization (train.py), due to the hardware
    limitation, it is not possible to train with all possible set of hyperparameters.
    Therefore, we use a set of randomized hyperparameters chosen with a ubiform
    distribution.
    
    The reason we do this approach is because of the high sensitivity of RL models
    to hyperparameters.
    
    We use the best hyperparameters extracted from MLP and use it in CNN. However,
    it is better to optimize CNN seperately but due to the limtation of the software we 
    just optimize the paramaeters of the MLP's network.
    
    Finally, for domain randomization we just define multiple distribution and we use them
    to optimize the network with hypereparameters extracted above. (we do not consider
    domain randomization optimization with this approach)
        
    '''
    def prepare_hyperparams(trial, params):
        hyperparams = {}
        search_space = search_spaces

        params_search_space = search_space[params['train_domain']]['ppo'].items()

        for param_name, value in params_search_space:
            hyperparams[param_name] = trial._suggest(param_name, value)
        return hyperparams 
    
    def optimize_agent(trial, params):
        hyperparams = prepare_hyperparams(trial, params=params)

        logs_dir = f"./experiments/experiments_{params['study_name']}/logs_{params['study_name']}"
        check_path(logs_dir)
        # initialize environments
        train_domain = params['train_domain']
        env_source = gym.make(f'CustomHopper-{train_domain}-v0')
        evaluate_domain = params['evaluate_domain']
        env_target = gym.make(f'CustomHopper-{evaluate_domain}-v0')
        
        model = PPO(env=env_source,
                    policy="MlpPolicy",
                    learning_rate=hyperparams['learning_rate'],
                    gamma=hyperparams['gamma'],
                    clip_range=hyperparams['clip_range'],
                    ent_coef=hyperparams['ent_coef'],
                    gae_lambda=hyperparams['gae_lambda'],
                    verbose=0,
                    device="cuda",
                    tensorboard_log=logs_dir)

        # initialize evaluation callback
        eval_callback = TrialEvalCallback(env_target, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ,
                                      deterministic=True)

        model.learn(total_timesteps=N_TIMESTEPS, progress_bar=True, callback=eval_callback)
        mean_reward = eval_callback.last_mean_reward
        # save best model
        if len(study.trials) == 1:
            model.save(f"./experiments/experiments_{params['study_name']}/best_model/hopper")

        elif mean_reward > study.best_trial.value:
            print('here')
            model.save(f"./experiments/experiments_{params['study_name']}/best_model/hopper")   
        del model
        env_source.close()
        env_target.close()
        
        return mean_reward
    
    n_trials = 50
    eperiment_path = f"./experiments/experiments_{params['study_name']}"
    check_path(eperiment_path)
    storage_url = f"sqlite:///{eperiment_path}/{params['study_name']}.db"
    study = optuna.create_study(direction='maximize', storage=storage_url, study_name=params['study_name'])
    experiment = partial(optimize_agent, params=params)
    study.optimize(experiment, n_trials=n_trials, show_progress_bar=True)
    
    print("Top 10 trials")
    saved_study = optuna.load_study(study_name=params['study_name'], storage=storage_url)
    trials = sorted(saved_study.trials, key=lambda t:t.value, reverse=True)
    for i in range(10):
        print("Trial number:", trials[i].number)
        print("Trial value", trials[i].value)
        print("Trial hyperparameters", trials[i].params)
        print()
            
def run_hpo(params):
    print("Running optuna tuning with params:", params)
    main(params)