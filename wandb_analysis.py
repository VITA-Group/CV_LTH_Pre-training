# %%
import pandas as pd 
import wandb
import matplotlib.pyplot as plt


api = wandb.Api()


entity, project = '828w', 'low_data_number_of_samples'

# %%

# Code

def get_runs_df(entity, project):

    runs =  api.runs(f'{entity}/{project}')

    summary_list, config_list, name_list = [], [], []

    for run in runs:

        summary_list.append(run.summary._json_dict)
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
             if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
        })



    runs_df['best_SA'] = runs_df['summary'].apply(lambda x: x['best_SA'] if 'best_SA' in x else None)


    runs_df['remain_weight'] = runs_df['summary'].apply(lambda x: int(x['remain_weight_0']) if 'remain_weight_0' in x else None)


    runs_df['train_size'] = runs_df['summary'].apply(lambda x: x['train_size'] if 'train_size' in x else None)

    runs_df['mask_dir'] = runs_df['config'].apply(lambda x: x['mask_dir'] if 'mask_dir' in x else None)


    runs_df['dataset'] = runs_df['config'].apply(lambda x: x['dataset'] if 'dataset' in x else None)


    runs_df["few_shot_ratio"] = runs_df['config'].apply(lambda x: x['few_shot_ratio'] if 'few_shot_ratio' in x else None)
    
    runs_df['arch'] = runs_df['config'].apply(lambda x: x['arch'] if 'arch' in x else None)

    runs_df["stripped_name"] = runs_df["mask_dir"].apply(lambda x: x.split("/")[-1])

    
    
    return runs_df

entity = "828w"

projects_with_number_of_samples = ["downstream_v3", "downstream_v4", "low_data_number_of_samples"]
projects_with_few_shot_ratio = ["downstream_v2"]

all_possible_runs_df = pd.DataFrame()

for project in projects_with_number_of_samples:
    runs_df = get_runs_df(entity, project)
    print(len(runs_df))
    all_possible_runs_df = pd.concat([all_possible_runs_df, runs_df])
    all_possible_runs_df["sample_type"] = "number_of_samples"
print("added number of samples")
print(len(all_possible_runs_df))

for project in projects_with_few_shot_ratio:
    runs_df = get_runs_df(entity, project)
    print(len(runs_df))
    runs_df["sample_type"] = "few_shot_ratio"
    all_possible_runs_df = pd.concat([all_possible_runs_df, runs_df])
print("added few shot ratio")
print(len(all_possible_runs_df))


# get the dictionary mask_dir_to_remain_weight from all possible runs


# %%

mask_dir_to_remain_weight = {}

for i, row in all_possible_runs_df.iterrows():
    if row["stripped_name"] not in mask_dir_to_remain_weight and row["stripped_name"] is not None and row["remain_weight"]:
        # if row["remain_weight"]  is not nan
        if row["remain_weight"] > 0:
            mask_dir_to_remain_weight[row["stripped_name"]] = row["remain_weight"]


all_possible_runs_df["normalized_remain_weight"] = all_possible_runs_df["stripped_name"].apply(lambda x: mask_dir_to_remain_weight[x] if x in mask_dir_to_remain_weight else None)

# drop rows where normalized_remain_weight is None

all_possible_runs_df = all_possible_runs_df[all_possible_runs_df["normalized_remain_weight"].notna()]


# %%
# drop failed runs

all_possible_runs_df = all_possible_runs_df[all_possible_runs_df["best_SA"].notna()]
print(len(all_possible_runs_df))
# %%

# for runs with few_shot_ratio


few_shot_df = all_possible_runs_df[all_possible_runs_df["sample_type"] == "few_shot_ratio"]

# sort by arch, dataset, few_shot_ratio

few_shot_df = few_shot_df.sort_values(by=["arch", "dataset", "few_shot_ratio"])


# %%

# drop where few_shot_ratio is None

few_shot_df = few_shot_df[few_shot_df["few_shot_ratio"].notna()]

for arch in few_shot_df["arch"].unique():
    print(arch)
    arch_df = few_shot_df[few_shot_df["arch"] == arch]
    for dataset in arch_df["dataset"].unique():
        dataset_df = arch_df[arch_df["dataset"] == dataset]
        # sort by normalized_remain_weight
        dataset_df = dataset_df.sort_values(by=["normalized_remain_weight"])
        for remaining_weights in dataset_df["normalized_remain_weight"].unique():
            
            remaining_weights_df = dataset_df[dataset_df["normalized_remain_weight"] == remaining_weights]
            # sort by few_shot_ratio
            remaining_weights_df = remaining_weights_df.sort_values(by=["few_shot_ratio"])
            plt.plot(remaining_weights_df["few_shot_ratio"], remaining_weights_df["best_SA"], label=remaining_weights)
        # legend lower right
        plt.legend(loc="lower right")
        plt.title(f"{arch} {dataset}")
        # x axis "Dataset Ratio"
        plt.xlabel("Dataset Ratio")
        # y axis "Validation Accuracy"
        plt.ylabel("Validation Accuracy")

        plt.show()

print("________________")
print("________________")
print("________________")
# %%


# for runs with number_of_samples

number_of_samples_df = all_possible_runs_df[all_possible_runs_df["sample_type"] == "number_of_samples"]

print(len(number_of_samples_df))
# sort by arch, dataset, few_shot_ratio
# %%
print("__________________")
print("number of samples")
number_of_samples_df = number_of_samples_df.sort_values(by=["arch", "dataset", "train_size"])

for arch in number_of_samples_df["arch"].unique():
    print(arch)
    arch_df = number_of_samples_df[number_of_samples_df["arch"] == arch]
    for dataset in arch_df["dataset"].unique():
        dataset_df = arch_df[arch_df["dataset"] == dataset]
        # sort by normalized_remain_weight
        dataset_df = dataset_df.sort_values(by=["normalized_remain_weight"])
        for remaining_weights in dataset_df["normalized_remain_weight"].unique():
            
            remaining_weights_df = dataset_df[dataset_df["normalized_remain_weight"] == remaining_weights]
            # sort by few_shot_ratio
            remaining_weights_df = remaining_weights_df.sort_values(by=["train_size"])
            plt.plot(remaining_weights_df["train_size"], remaining_weights_df["best_SA"], label=remaining_weights)
        # legend lower right
        plt.legend(loc="lower right")
        plt.title(f"{arch} {dataset}")
        # x axis "Dataset Ratio"
        plt.xlabel("N of Samples")
        # y axis "Validation Accuracy"
        plt.ylabel("Validation Accuracy")

        plt.show()
# %%
