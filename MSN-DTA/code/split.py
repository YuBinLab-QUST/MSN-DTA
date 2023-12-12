import pandas as pd
import argparse

def create_fold(df, fold_seed, frac):

    train_frac, val_frac, test_frac = frac
    test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(
        frac=val_frac / (1 - test_frac), replace=False, random_state=1
    )
    train = train_val[~train_val.index.isin(val.index)]

    return {
        "train": train.reset_index(drop=True),
        "valid": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def create_fold_setting_cold(df, fold_seed, frac, entities):

    if isinstance(entities, str):
        entities = [entities]

    train_frac, val_frac, test_frac = frac

    test_entity_instances = [
        df[e]
        .drop_duplicates()
        .sample(frac=test_frac, replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]


    test = df.copy()
    for entity, instances in zip(entities, test_entity_instances):
        test = test[test[entity].isin(instances)]

    if len(test) == 0:
        raise ValueError(
            "No test samples found. Try another seed, increasing the test frac or a "
            "less stringent splitting strategy."
        )


    train_val = df.copy()
    for i, e in enumerate(entities):
        train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

    val_entity_instances = [
        train_val[e]
        .drop_duplicates()
        .sample(frac=val_frac / (1 - test_frac), replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]
    val = train_val.copy()
    for entity, instances in zip(entities, val_entity_instances):
        val = val[val[entity].isin(instances)]

    if len(val) == 0:
        raise ValueError(
            "No validation samples found. Try another seed, increasing the test frac "
            "or a less stringent splitting strategy."
        )

    train = train_val.copy()
    for i, e in enumerate(entities):
        train = train[~train[e].isin(val_entity_instances[i])]

    return {
        "train": train.reset_index(drop=True),
        "valid": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }

dataset_name = 'davis'

SEED = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default=dataset_name)
    parser.add_argument("--SEED", type=int, default=SEED)

    args = parser.parse_args()

    df = pd.read_csv("../Data/" + args.dataset_name + ".csv")

    cold_target_fold = create_fold_setting_cold(df, args.SEED, [0.8, 0.1, 0.1], ['target_key'])
    cold_target_fold["train"].to_csv("../Data/" + args.dataset_name + "_cold_traget_train_"+str(args.SEED)+".csv", index=False)
    cold_target_fold["valid"].to_csv("../Data/" + args.dataset_name + "_cold_traget_valid_"+str(args.SEED)+".csv", index=False)
    cold_target_fold["test"].to_csv("../Data/" + args.dataset_name + "_cold_traget_test_"+str(args.SEED)+".csv", index=False)
    print(args.dataset_name+ " cold_target_fold done!  the shape of train, valid, test are: ", cold_target_fold["train"].shape, cold_target_fold["valid"].shape, cold_target_fold["test"].shape)

    cold_drug_fold = create_fold_setting_cold(df, args.SEED, [0.8, 0.1, 0.1], ['compound_iso_smiles'])
    cold_drug_fold["train"].to_csv("../Data/" + args.dataset_name + "_cold_drug_train_"+str(args.SEED)+".csv", index=False)
    cold_drug_fold["valid"].to_csv("../Data/" + args.dataset_name + "_cold_drug_valid_"+str(args.SEED)+".csv", index=False)
    cold_drug_fold["test"].to_csv("../Data/" + args.dataset_name + "_cold_drug_test_"+str(args.SEED)+".csv", index=False)
    print(args.dataset_name+ " cold_drug_fold done!  the shape of train, valid, test are: ", cold_drug_fold["train"].shape, cold_drug_fold["valid"].shape, cold_drug_fold["test"].shape)

    cold_target_drug_fold = create_fold_setting_cold(df, args.SEED, [0.8, 0.1, 0.1], ['target_key', 'compound_iso_smiles'])
    cold_target_drug_fold["train"].to_csv("../Data/" + args.dataset_name + "_cold_target_drug_train_"+str(args.SEED)+".csv", index=False)
    cold_target_drug_fold["valid"].to_csv("../Data/" + args.dataset_name + "_cold_target_drug_valid_"+str(args.SEED)+".csv", index=False)
    cold_target_drug_fold["test"].to_csv("../Data/" + args.dataset_name + "_cold_target_drug_test_"+str(args.SEED)+".csv", index=False)
    print(args.dataset_name+ " cold_target_drug_fold done!  the shape of train, valid, test are: ", cold_target_drug_fold["train"].shape, cold_target_drug_fold["valid"].shape, cold_target_drug_fold["test"].shape)
