import RandomForest
import Tree
import csv
import random
import functions

random.seed(5)
datasets = list()

with open('mushrooms.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    dataset = [row for row in reader]
    dataset = dataset[1:]
    datasets.append(dataset)

with open('nursery.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    dataset = [row for row in reader]
    for row in dataset:
        aux = row[0]
        row[0] = row[len(row) - 1]
        row[len(row) - 1] = aux
    datasets.append(dataset)

with open('connect-4.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    dataset = [row for row in reader]
    for row in dataset:
        aux = row[0]
        row[0] = row[len(row) - 1]
        row[len(row) - 1] = aux
    datasets.append(dataset)

"""ESPERIMENTO ALBERI DI DECISIONE."""
print('\n\n\nESPERIMENTO ALBERI DI DECISIONE\n\n\n')
datasets_names = ['MUSHROOMS', 'nursery', 'CONNECT-4']
num_features_t = [100, 50, 100]
depth_t = [100, 100, 100]
d_index_t = 0
for dataset in datasets:

    print('ANALISI SU DATASET:', datasets_names[d_index_t], '\n\n')

    dim_trainingset = int(len(dataset) * 2 / 3)
    shuffled_dataset = dataset
    random.shuffle(shuffled_dataset)
    training_set = [row for row in shuffled_dataset[0:dim_trainingset]]
    test_set = [row for row in shuffled_dataset[dim_trainingset + 1:]]
    print('LENGTH DATASET:', len(dataset), 'LENGTH TRAININGSET:', len(training_set), 'LENGTH TESTSET:', len(test_set),
          '\n')
    not_used_features = [var for var in range(1, len(dataset[0]))]

    my_tree = Tree.DTree(training_set, not_used_features, num_features_t[d_index_t], depth_t[d_index_t])
    test_score_t = 0
    for row in test_set:
        label, prob = my_tree.classify(row)
        if label == row[0]:
            test_score_t += 1

    test_score_t = test_score_t / len(test_set)
    print('SCORE SUL TESTSET: ', test_score_t, '\n\n\n')
    d_index_t += 1


"""ESPERIMENTO RANDOM FOREST."""
print('\n\n\nESPERIMENTO RANDOM FOREST\n\n\n')
datasets_names = ['MUSHROOMS', 'nursery', 'CONNECT-4']
num_features = [[10], [5], [10]]
depth = [[10, 20, 30], [5, 10, 15], [30, 20, 10]]
d_index = 0
for dataset in datasets:
    print('ANALISI SU DATASET:', datasets_names[d_index], '\n\n')

    dim_trainingset = int(len(dataset) * 2/3)
    shuffled_dataset = dataset
    random.shuffle(shuffled_dataset)
    training_set = [row for row in shuffled_dataset[0:dim_trainingset]]
    test_set = [row for row in shuffled_dataset[dim_trainingset+1:]]

    print('LENGTH DATASET:', len(dataset), 'LENGTH TRAININGSET:', len(training_set), 'LENGTH TESTSET:', len(test_set), '\n')

    not_used_features = [var for var in range(1, len(dataset[0]))]
    """Nei casi da noi considerati (classification su dataset le cui features sono categoriche) è più interessante
       ottimizzare su numero di feature considerate e profondità degli alberi piuttosto che sul numero di alberi
       della random forest."""

    num_trees = 10
    k = 5
    print('CROSSVALIDATION SUI PARAMETRI DELLA RANDOM FOREST:\n')
    scores = list()
    params = list()
    d_num_features = num_features[d_index]
    d_depth = depth[d_index]
    for num in d_num_features:
        for d in d_depth:
            print('PARAMETRI: ', 'folds =', k, ', num_features =', num, ', depth =', d)
            score = functions.k_fold_cross_validation(training_set, k, num_trees, not_used_features, num, d)
            print('Score:', score, '\n')
            scores.append(score)
            params.append([num, d])

    best_score = max(scores)
    best_params = params[scores.index(best_score)]
    print('\n')
    print('BEST NUM_FEATURES: ', best_params[0], 'BEST DEPTH: ', best_params[1], 'BEST SCORE: ', best_score, '\n')

    my_forest = RandomForest.DRandomForest(num_trees, training_set, not_used_features, best_params[0], best_params[1])
    test_score = 0
    for row in test_set:
        label = my_forest.classify(row)
        if label == row[0]:
            test_score += 1

    test_score = test_score / len(test_set)
    print('SCORE SUL TESTSET: ', test_score, '\n\n\n')
    d_index += 1
