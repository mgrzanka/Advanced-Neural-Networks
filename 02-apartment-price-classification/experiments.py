import copy
from train import train


def generate_experiments():
    baseline = {
        "epochs": 100,
        "dropout_p": 0.3,
        "weight_decay": 1e-4,
        "network_sizes": [128, 64],
        "use_batch_norm": True,
        "use_one_hot": False,
        "use_weighted_criterion": True,
        "experiment_name": "00_Baseline"
    }

    experiments = [baseline]

    # 1. Batch Norm
    exp_bn_off = copy.deepcopy(baseline)
    exp_bn_off["use_batch_norm"] = False
    exp_bn_off["experiment_name"] = "01_BatchNorm_Off"
    experiments.append(exp_bn_off)

    # 3. Dropout
    for p in [0.0, 0.5]:
        exp_drop = copy.deepcopy(baseline)
        exp_drop["dropout_p"] = p
        exp_drop["experiment_name"] = f"02_Dropout_{p}"
        experiments.append(exp_drop)

    # 4. L2 (Weight Decay)
    for wd in [0.0, 1e-2]:
        exp_wd = copy.deepcopy(baseline)
        exp_wd["weight_decay"] = wd
        exp_wd["experiment_name"] = f"03_WeightDecay_{wd}"
        experiments.append(exp_wd)

    # 5. Model size (capacity)
    sizes = {
        "Underfit": [32, 16],
        "Overfit": [1024, 512, 256]
    }
    for name, size in sizes.items():
        exp_size = copy.deepcopy(baseline)
        exp_size["network_sizes"] = size
        exp_size["experiment_name"] = f"04_NetworkSize_{name}"
        experiments.append(exp_size)

    # 6. Encoding (One-Hot)
    exp_ohe = copy.deepcopy(baseline)
    exp_ohe["use_one_hot"] = True
    exp_ohe["experiment_name"] = "05_Encoding_OneHot"
    experiments.append(exp_ohe)

    # 7. Weighted criterion
    exp_unweighted = copy.deepcopy(baseline)
    exp_unweighted["use_weighted_criterion"] = False
    exp_unweighted["experiment_name"] = "06_Unweighted_Criterion"
    experiments.append(exp_unweighted)

    return experiments

if __name__ == '__main__':
    experiments_to_run = generate_experiments()
    print(f"Rozpoczynam sekwencję {len(experiments_to_run)} eksperymentów.")

    for i, config in enumerate(experiments_to_run):
        print(f"\n--- Experiment {i+1}/{len(experiments_to_run)}: {config['experiment_name']} ---")
        train(config)

    print("\nAll experments finished. Results in Weights & Biases panel.")
