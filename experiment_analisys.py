import optuna

storage_url = "sqlite:///target_target.db"
study_name = "target_target"

saved_study = optuna.load_study(study_name=study_name, storage=storage_url)
trials = sorted(saved_study.trials, key=lambda t:t.value, reverse=True)
for i in range(10):
    print("Trial number:", trials[i].number)
    print("Trial value", trials[i].value)
    print("Trial hyperparameters", trials[i].params)
    print()