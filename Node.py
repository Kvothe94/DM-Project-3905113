"""Classe che rappresenta i nodi dell'albero di decisione.
   contiene diverse variabili utilizzate per l'addestramento
   e la gestione dell'albero."""


class DTreeNode:
    def __init__(self):
        # Variabile che indica la classe output del nodo (se presente nodo foglia, altrimenti intermedio).
        self.label = None

        # Variabile che indica la feature trattata in questo nodo.
        self.feature = None

        # Variabile che contiene una lista di nodi successivi in base al valore della feature indicata nella
        # variabile feature.
        self.nexts = dict()

        # Parte del dataset che viene usato nel particolare nodo.
        self.dataset = None

        # Variabile in cui, nel caso di un nodo foglia, memorizziamo la probabilità della label indicata
        self.probability = None
