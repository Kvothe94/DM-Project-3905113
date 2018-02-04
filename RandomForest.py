import Tree

class DRandomForest:

    def __init__(self, num_trees, dataset, not_used_features, num_features, max_depth):
        self.trees = list()
        for i in range(num_trees):
            aux_not_used_features = not_used_features.copy()
            aux_tree = Tree.DTree(dataset, aux_not_used_features, num_features, max_depth)
            self.trees.append(aux_tree)

    def classify(self, sample):

        classifications = list()
        for tree in self.trees:
            label, probability = tree.classify(sample)
            classifications.append(label)

        classes = {var: 0 for var in classifications}
        for cls in classifications:
            classes[cls] = classes[cls] + 1

        n_class = max(list(classes.values()))
        n_class = list(classes.values()).index(n_class)
        best_class = list(classes.keys())[n_class]
        return best_class
