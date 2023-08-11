import os
from functools import partial
import neptune
import neptune.integrations.optuna as optuna_utils
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_hopper import *
from stable_baselines3.common.monitor import Monitor
import optuna
from optuna_search_spaces import search_spaces, search_spaces_udr
from policy_gradient import *

N_EVALUATIONS = 5
N_TIMESTEPS = 100_000
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 100


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


def prepare_hyperparams(trial, params):
    hyperparams = {}
    search_space = search_spaces
    if params['udr_enable']:
        search_space = search_spaces_udr

    params_search_space = search_space[params['train_domain']]['ppo'].items()

    for param_name, value in params_search_space:
        hyperparams[param_name] = trial._suggest(param_name, value)
    return hyperparams


def experiment_fn(trial, study, params):
    hyperparams = prepare_hyperparams(trial, params)


    # initialize environments
    train_domain = params['train_domain']
    env_source = gym.make(f'CustomHopper-{train_domain}-v0')
    env_source.seed(0)
    evaluate_domain = params['evaluate_domain']
    env_target = gym.make(f'CustomHopper-{evaluate_domain}-v0')
    env_target.seed(0)
    # if params['udr_enable']:
    #     masses_to_set = env_source.get_parameters()[1:]
    #     lower_bound = masses_to_set * (1-0.5*hyperparams['eps_udr'])
    #     upper_bound = masses_to_set * (1+0.5*hyperparams['eps_udr'])
    #     distribution_set = [lower_bound[0], upper_bound[0], lower_bound[1], upper_bound[1], lower_bound[2], upper_bound[2]]
    #     env_source.set_udr_training(True)
    #     env_source.set_udr_distribution(distribution_set)
    #     hyperparams.pop('eps_udr')

    env_source.reset()

    directory_experiments = f"./experiments_{study.study_name}/trial_{trial.number}"

    # initialize evaluation callback
    eval_callback = TrialEvalCallback(env_target, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ,
                                      deterministic=True)

    model = PPO(env=env_source,
                policy='MlpPolicy',
                learning_rate=hyperparams['learning_rate'],
                gamma=hyperparams['gamma'],
                clip_range=hyperparams['clip_range'],
                ent_coef=hyperparams['ent_coef'],
                gae_lambda=hyperparams['gae_lambda'],
                verbose=0,
                device="cuda",
                tensorboard_log=directory_experiments)
    
    model.set_random_seed(0)

    nan_encountered = False

    try:
        model.learn(N_TIMESTEPS, progress_bar=True, callback=eval_callback)

    except AssertionError:
        nan_encountered = True

    finally:
        env_source.close()

    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    # save best model
    if len(study.trials) == 1:
        model.save(f"./experiments_{study.study_name}/best_model/hopper")

    elif eval_callback.last_mean_reward > study.best_trial.value:
        print('here')
        model.save(f"./experiments_{study.study_name}/best_model/hopper")

    # return the mean reward tested on the evaluation environment
    return eval_callback.last_mean_reward

def run_hpo(params):
    run = neptune.init_run(
        project="gianmarcobll/amlHopper",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhNjA4ZDMwYS1jMWI2LTRjYjUtYmY1Ny1hZWIzMGRhZmVjZDQifQ==",
    )   
    neptune_callback = optuna_utils.NeptuneCallback(run)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(), direction="maximize", study_name=params['study_name'],
        storage='sqlite:///experiment.db',
        pruner=optuna.pruners.SuccessiveHalvingPruner()
    )

    experiment = partial(experiment_fn, study=study, params=params)
    study.optimize(experiment, n_trials=50, callbacks=[neptune_callback])
    run.stop()
