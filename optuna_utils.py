from optuna import Trial


def create_trial_from_dict(trial: Trial, params: dict) -> Trial:
    for k, v in params.items():
        if v["type"] == "int":
            trial.suggest_int(k, int(v["min"]), int(v["max"]))
        elif v["type"] == "float":
            trial.suggest_float(k, float(v["min"]), float(v["max"]), log=v["log"])
        elif v["type"] == "list":
            trial.suggest_categorical(k, v["values"])
        else:
            raise ValueError(f"Unknown parameter type: {v['type']}")

    return trial
