# DM-Project-3905113
Progetto per il corso di Data Mining.

Il progetto è stato sviluppato in ambiente
Anaconda 3 con python 3.6.

Non vengono utilizzate librerie non standard,
pertanto il progetto dovrebbe funzionare senza
installare librerie particolari.

Per replicare l'esperimento dovrebbe bastare
eseguire il codice presente in main.py.

N.B: Lo studio sul dataset connect-4.csv richiede
     un certo tempo per l'ottimizzazione dei
     parametri; se si vuole vedere i risultati
     del codice senza eseguirlo si può fare riferimento
     al file result.txt che contiene la copia dell'
     output della console di output di un esecuzione
     del programma.
     
Lo scopo del progetto è fornire un algoritmo per l'utilizzo
di Random Forest e Decision Tree applicati su dataset
orientati alla classificazione con features categoriche.

Il progetto è strutturato come segue:

- DTreeNode: classe che rappresenta i nodi dell'albero, contiene gli attributi necessari per la memorizzazione del modello (Node.py).
- DTree: classe che contiene i metodi e gli attributi per la creazione e la gestione dell'albero nonchè i metodi necessari per la
         classificazione (Tree.py).
- DRandomForest: classe che contiene i metodi e gli attributi per la creazione e la gestione della random forest nonchè i metodi
                 necessari per la classificazione (RandomForest.py).

A tali classi si aggiungono alcune funzioni necessarie per la selezione dei parametri tramite k-fold cross validation (functions.py).


Il dataset fornito in input agli algoritmi, oltre a rispettare
le suddette condizione deve essere caratterizzato nel seguente
modo:
  
  1) Ogni sample deve corrispondere a una RIGA del dataset.
  
  2) Ogni feature deve corrispondere a una COLONNA del sample.
  
  3) La CLASSE sulla quale si vuole svolgere la classificazione
     deve essere la PRIMA features di ogni sample.
    
  N.B: Non tutti i dataset utilizzati nel programma rispettavano
       le suddette condizione pertanto all'inizio di main.py è
       presente del codice che porta i dataset nel formato atteso.
       
Le condizioni di terminazione sugli alberi sono date da un parametro
che ne limita la depth massima. Non viene effettuato alcun altro genere
di pruning.

Nel main.py vengono effettuati i seguenti esperimenti:

  1) Nella prima parte per ogni dataset considerato viene
     addestrato un albero di decisione senza limitarne la
     profondità o il numero di features e ne vengono
     presentate le performance in termini di percentuale
     del testset classificato correttamente.
     
  2) Nella seconda parte per ogni dataset viene determinata
     tramite cross validation il migliore valore di depth per
     gli alberi della random forest; dopodichè viene addestrata
     la suddetta random forest e ne vengono presentate le performance
     in maniera analoga alla prima parte.
     
     N.B: Non vengono considerati altri paramentri per la cross validation
          a causa di problemi di complessità computazionale: il programma
          risulterebbe troppo lento per una fruizione immediata. Ciònonostante
          bastano alcune piccole modifiche nel codice per andare a considerare
          anche altri parametri.
          La scelta di ottimizzare la depth è stata fatta dopo aver notato
          sperimentalmente che gli altri parametri non modificano le performance
          delle random forest in maniera significativa.
          
Studiando i dati ottenuti dagli esperimenti effettuati si notano le seguenti cose:

  1) Per quanto riguarda mushroom.csv e nursery.csv si può notare che l'incremento delle performance dato passando da Decision 
     Tree a Random Forest non è percepibile (circa 0 %).
     Nel caso del dataset connect-4.csv invece si inizia a vedere un incremento delle performance nel passaggio tra i due modelli
     (circa 6 %).
     
  2) Analizzando tramite debugger la struttura dell'albero generato per il dataset
     mushroom.csv si può notare che il numero di feature analizzate risulta essere
     piuttosto basso => La "regola" di classificazione risulta dipendere da poche
     features (come ci si potrebbe aspettare ragionando in termini del problema reale).
       
Considerando i risultati ottenuti da un punto di vista teorico e ricordando che Decision
Tree e Random Forest basano il loro funzionamento sul principio secondo cui esiste un qualche
genere di "regola" basata sulle caratteristiche delle features per la classificatione, possiamo
pensare di trarre la seguente conclusione:

Nel caso di dataset categorici con feature categoriche risulta possibile, per gli algoritmi considerati,
estrarre, con più o meno successo, la "regola" che caratterizza la classificazione dei samples nel
dataset.
Considerando i risultati sui tre dataset si può notare che solo all'aumentare della complessità della "regola"
di classificazione del dataset le performance di Decision Tree e Random Forest divergono e diventano
più favorevoli nel caso delle Random Forest.





       
       
       
