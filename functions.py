import random
import RandomForest
"""File contenente le funzioni necessarie per la gestione e per la creazione dell'albero."""


def k_fold_cross_validation(dataset, k, num_trees, not_used_features, num_features, max_depth):

    datasets = split_cross(dataset, k)
    scores = list()
    for fold in datasets:

        validationset = fold

        trainingset = list()
        for data in datasets:
            if data != fold:
                for row in data:
                    trainingset.append(row)

        copy_not_used_features = not_used_features.copy()
        r_forest = RandomForest.DRandomForest(num_trees, trainingset, copy_not_used_features, num_features, max_depth)
        score = 0
        for sample in validationset:
            if sample[0] == r_forest.classify(sample):
                score += 1
        score = score / len(validationset)
        scores.append(score)

    mean_score = sum(scores) / len(scores)
    return mean_score


def split_cross(dataset, k):

    datasets = list()
    dataset_copy = dataset.copy()
    len_fold = int(len(dataset) / k)
    for i in range(k):
        if len(dataset_copy) > len_fold:
            aux = random.choices(dataset_copy, k=len_fold)
        else:
            aux = dataset_copy
        datasets.append(aux)

    return datasets

