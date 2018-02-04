import Node
import math
import random

"""Classe che contiene i dati e i metodi necessari per il funzionamento del decision tree."""


class DTree:

    def __init__(self, dataset, not_used_features, num_features, max_depth):

        self.root = Node.DTreeNode()
        self.root.dataset = dataset
        depth = 0
        
        class_values = [value for value in [row[0] for row in self.root.dataset]]
        prob_class = {var: 0 for var in class_values}

        if len(prob_class) == 1:
            labels = list(prob_class.keys())
            self.root.label = labels[0]
            self.root.probability = 1.0
            return

        tot_value = len(self.root.dataset)
        for value in class_values:
            prob_class[value] = prob_class[value] + 1

        for value in prob_class:
            prob_class[value] = prob_class[value] / tot_value

        labels = list(prob_class.keys())
        prob = list(prob_class.values())
        best_class = labels[prob.index(max(prob))]
        self.root.label = best_class
        self.root.probability = max(prob)

        if len(not_used_features) == 0:
            return

        features = list()
        if num_features < len(not_used_features):
            features = random.sample(not_used_features, num_features)
        else:
            features = not_used_features

        datasets_list, best_feature = split(self.root.dataset, features)
        not_used_features.remove(best_feature)
        self.root.feature = best_feature
        for datas in datasets_list:
            new_node = Node.DTreeNode()
            new_node.dataset = datasets_list[datas]
            aux_not_used_features = not_used_features.copy()
            self.root.nexts.update({datas: build_node(new_node, aux_not_used_features, num_features, max_depth, depth)})

        return

    def dim_tree(self):
        dim = 1
        nexts_keys = list(self.root.nexts.keys())
        for key in nexts_keys:
            dim = dim + dim_node(self.root.nexts[key])
        return dim

    def classify(self, sample):

        if self.root.feature is None:
            return self.root.label, self.root.probability

        feature_value = sample[self.root.feature]
        if feature_value in list(self.root.nexts.keys()):
            next_node = self.root.nexts[feature_value]
            return classify_node(next_node, sample)
        else:
            return self.root.label, self.root.probability


def compute_cost(dataset, feature):

    tot_value = len(dataset)

    """Calcoliamo l'entropia del nostro dataset (in riferimento alle classi di output.)"""
    class_values = [value for value in [row[0] for row in dataset]]
    prob_class = {var: 0 for var in class_values}
    for value in class_values:
        prob_class[value] = prob_class[value] + 1

    for value in prob_class:
        prob_class[value] = prob_class[value] / tot_value

    prev_entropy = 0
    for value in prob_class:
        if prob_class[value] != 0:
            prev_entropy = prev_entropy - prob_class[value] * math.log2(prob_class[value])

    """Calcoliamo l'entropia del nostro dataset dopo aver fatto lo split sulla base della
       feature passata come argomento."""
    # Innanzitutto calcoliamo la probabilità del valore della feature per ogni valore.
    feature_values = [value for value in [row[feature] for row in dataset]]
    prob_values = {var: 0 for var in feature_values}
    for value in feature_values:
        prob_values[value] = prob_values[value] + 1

    for value in prob_values:
        prob_values[value] = prob_values[value] / tot_value

    # Dobbiamo quindi calcolare per ogni valore della feature l'entropia per quel valore della feature.
    num_feature_class = dict()
    for feature_value in prob_values:
        num_feature_class.update({feature_value: dict()})
        for class_value in prob_class:
            aux_var = [row for row in dataset if (row[0] == class_value and row[feature] == feature_value)]
            num_feature_class[feature_value].update({class_value: len(aux_var)})

    entropy_feature = dict()
    for feature_value in prob_values:
        entropy_feature.update({feature_value: 0})
        tot_sample = sum(num_feature_class[feature_value].values())

        for class_value in prob_class:
            prob_feature_class = num_feature_class[feature_value][class_value] / tot_sample

            if prob_feature_class != 0:
                entropy_feature[feature_value] = entropy_feature[feature_value] - \
                                                 prob_feature_class * math.log2(prob_feature_class)

    # A questo punto possiamo calcolare l'entropia del nostro dataset dopo aver fatto lo split
    # sulla feature considerata.
    post_entropy = 0
    for feature_value in prob_values:
        post_entropy = post_entropy + prob_values[feature_value] * entropy_feature[feature_value]

    # Ora possiamo calcolare l'Information Gain nel seguente modo.
    inf_gain = prev_entropy - post_entropy
    return inf_gain


def split(dataset, not_used_features):
    # Innanzitutto troviamo la feature che conviene usare per splittare il dataset
    # tra quelle fornite in not_used_features.
    feature_cost = [compute_cost(dataset, feature) for feature in not_used_features]
    best_cost = max(feature_cost)
    best_feature = not_used_features[feature_cost.index(best_cost)]

    # A questo punto vogliamo fare lo split del dataset in base alla feature trovata.
    datasets_list = dict()
    feature_values = [value for value in [row[best_feature] for row in dataset]]
    feature_values = {var: 0 for var in feature_values}

    for feature_value in feature_values:
        aux_var = [row for row in dataset if row[best_feature] == feature_value]
        datasets_list.update({feature_value: aux_var})

    return datasets_list, best_feature


def build_node(my_node, not_used_features, num_features, max_depth, depth):

    depth = depth + 1
    class_values = [value for value in [row[0] for row in my_node.dataset]]
    prob_class = {var: 0 for var in class_values}

    if len(prob_class) == 1:
        labels = list(prob_class.keys())
        my_node.label = labels[0]
        my_node.probability = 1.0
        return my_node

    tot_value = len(my_node.dataset)
    for value in class_values:
        prob_class[value] = prob_class[value] + 1

    for value in prob_class:
        prob_class[value] = prob_class[value] / tot_value

    labels = list(prob_class.keys())
    prob = list(prob_class.values())
    best_class = labels[prob.index(max(prob))]
    my_node.label = best_class
    my_node.probability = max(prob)

    if len(not_used_features) == 0 or depth >= max_depth:
        return my_node

    features = list()
    if num_features < len(not_used_features):
        features = random.sample(not_used_features, num_features)
    else:
        features = not_used_features

    datasets_list, best_feature = split(my_node.dataset, features)
    not_used_features.remove(best_feature)
    my_node.feature = best_feature
    for datas in datasets_list:
        new_node = Node.DTreeNode()
        new_node.dataset = datasets_list[datas]
        aux_not_used_features = not_used_features.copy()
        my_node.nexts.update({datas: build_node(new_node, aux_not_used_features, num_features, max_depth, depth)})

    return my_node


def dim_node(my_node):
    dim = 1
    nexts_keys = list(my_node.nexts.keys())
    for key in nexts_keys:
        dim = dim + dim_node(my_node.nexts[key])
    return dim


def classify_node(my_node, sample):

    if my_node.feature is None:
        return my_node.label, my_node.probability

    feature_value = sample[my_node.feature]
    if feature_value in list(my_node.nexts.keys()):
        next_node = my_node.nexts[feature_value]
        return classify_node(next_node, sample)
    else:
        return my_node.label, my_node.probability
