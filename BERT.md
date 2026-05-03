<script>
MathJax = {
  tex: {
    inlineMath: {'[+]': [['$', '$']]}
  }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@4/tex-chtml.js"></script>

# BERT

## Introduction

**BERT** signifie **Bidirectional Encoder Representation from Transformer**. C'est un LLM (Large Langage Model), un modèle de grande taille dédié au traitement du langage, décrit dans [cette publication](https://arxiv.org/abs/1810.04805). Il a vu le jour en 2018 (la même année que GPT)

**Bert** est un modèle **encoder only**, qui est entrainé sur des tâches de **Masked Langage Modeling** et de **Next Sentence Prédiction**. Il vise à apprendre à encoder des séquences de texte de façon très efficace pour ces tâches. Nous reviendrons plus loin sur ces points.

L'idée derrière **Bert** que l'apprentissage initial soit **bidirectionnel**, au sens ou le contexte extrait par attention puisse venir de mots situés plus loin dans la séquence d'entrée. Ainsi, Bert peut prédire des mots masqués un peu partout dans un texte, là ou GPT ne peut prédire que les mots suivants dans une séquence. C'est à ce titre que BERT est utilisé comme base des **ViT** en computer vision.

Avec un modèle Bert pré-entrainé, il est possible de l'utiliser dans des **downstream tasks** (voir la section [GPT](GPT.md)), à condition de remplacer le module en sortie de Bert, de le remplacer par un module correspondant à la tâche souhaitée, puis de "fine tuner" le module final.

Ici, on commencera par s'intéresser à l'architecture de BERT, qui présente quelques spécificités intéressantes.

## Architecture de BERT

Voici une représentation de l'architecture de BERT.

![architecture de BERT](Images/bert_architecture.png)

Comme on peut le voir dans l'image précédente, il s'agit d'un encodeur transformers classique.

Notons néanmoins un point spécifique qui le distingue du transformer originel :
le **positional encoding** est ici effectué par une simple couche dense :
la position d'un token dans la séquence est un entier, transformé en *one hot vector* de dimension $s$, et ce vecteur passe dans la couche linéaire pour donner un vecteur de la dimension de l'embedding $e$. Cette couche est apprise pendant l'entrainement.

L'embedding des tokens est réalisé par l'algorithme **wordpiece**, une variante de **BPE**.

### Quelques chiffres

Voyons quelques chiffres concernant les deux principales versions de BERT :
La taille du vocabulaire est, dans les deux cas, fixée à $d_{dict}=30000$.

- BERT Base :
  - dimension de l'embedding $e=768$.
  - taille des séquences $s = 512$
  - nombre de couches d'encodeur : $L=12$
  - nombre de têtes d'attention par couches : 12 Attention,
  - nombre de neurones dans les couches cachées des Feed : 3072
  - nombre total de paramètres : 110M

La version plus grande de Bert est caractérisée comme suit :

- BERT Large :
  - dimension de l'embedding $e=1024$.
  - taille des séquences $s = 512$
  - nombre de couches d'encodeur : $L=24$
  - nombre de têtes d'attention par couches : 16 Attention,
  - nombre de neurones dans les couches cachées des Feed : 4096
  - nombre total de paramètres : 340M


### Apprentissage

Pour décrire cet apprentissage, ma description se fera en 3 étapes séparées, mais Bert applique toutes ces étapes.

Bert est pré-entrainé à réaliser deux tâches en parallèle, que voici :

#### Masked Langage Modelisation (MLM)

La première tâche à laquelle Bert est entrainé est de reconstruire des phrases partiellement masquées.

Prenons la séquence (ici très courte) : `Les chats mangent des souris`

15% des token de la séquence vont être masqués en entrée, et remplacés par un
token spécial `[Mask]` ce qui pourrait donner la séquence `Les [Mask] mangent des souris`

L'objectif du réseau est de fournir, en sortie, un choix de token correct pour chaque token masqué. Ceci est fait comme illustré dans la figure suivante :

![Bert MLM](Images/Bert_MLM.png)

1. Les blocs d'attentions enrichissent chaque token de la séquence.
2. En sortie des transformers, les tokens `[Mask]` sont donc enrichis (en violet).
3. Le DNN est en charge de choisir, pour chaque token enrichi, le mot le plus probable (en bleu). 

Cette tâche peut être considérée comme **non supervisée**, au sens ou l'on peut apprendre sur des données non labelisées (n'importe quel texte, de n'importe quelle origine).

Ici, on peut noter la présence d'un token spécial `[CLS]`, placé en début de séquence, et sans intérêt pour cette tâche, mais qui sera utilisé pour la prochaine.

#### Next Sentence Prediction (NSP)

Ici, l'idée est d'entrainer le modèle, toujours de façon **non supervisée**, à saisir la relation entre deux phrases. Pour cela, une séquence d'entrée va être composée de deux portions de texte tirées au hasard dans le corpus. Dans 50% des cas, la seconde portion est effectivement la phrase suivante de la premiere  dans le corpus, dans les autres cas, c'est une portion qui n'a rien a voir. En voici deux exemples :

- `[CLS] Les chats mangent des souris. [SEP] Ils jouent souvent avec avant de les manger.`
- `[CLS] Les chats mangent des souris. [SEP] Ces couches de neurones sont linéaires.`

L'objectif du modèle est de déterminer si la seconde phrase est suit effectivement la première. C'est une tâche de **classification binaire** (oui ou non).

Pour cela, **un seul token en sortie** est utilisé pour représenter toute la séquence. Il s'agit du token correspondant à `[CLS`. Il est injecté dans une couche dense avec softmax, qui fournit la réponse attendue (oui ou non).

La partie DNN (en vert dans la figure précédente) est donc composée de 2 réseaux indépendants, travaillant sur des portions différentes  :

1. Une couche de taille $e \times d_{dict}$ qui prédit le token suivant pour chaque token de la séquence après `[CLS]`
2. Une couche de taille $e \times 2$ qui prédit si oui ou non la seconde phrase est la suite de la premiere. Cette couche ne travaille qu'avec le token `[CLS]`, enrichi par le contexte de toute la séquence.

Selon les auteurs, le pré-training de Bert à l'aide de cette NSP non supervisée (pourtant très simple) améliore grandement les performances du réseau lors du fine tuning des *downstream tasks* (voir ci-dessous)

Selon les auteurs, l'idée est que le token `[CLS]` soit, en sortie de l'encoder, une **représentation de la totalité de la séquence**.

### Entrainement des downstream tasks

Pour toute tâche en aval (*Downstream task*), il est nécessaire de supprimer la couche finale de BERT, et de la remplacer par une couche adaptée au problème à traiter. Ces problèmes peuvent être de deux types :

- Pour toute tâche de **niveau token**, on utilise la sortie de l'encodeur correspondant à la séquence complète, privée de `[CLS]`, qu'on passe à la nouvelle couche.
- Pour toute tâche de type **classification** (comme l'analyse de sentiments), on utilise la sortie de l'encodeur correspondant au token `[CLS]`, qu'on passe à la nouvelle couche.

On peut alors procéder au **fine tuning** du modèle, sur des données **labelisées**. *Je n'ai pas compris si on fine tunait tout le modèle ou seulement la nouvelle tête de sortie. A priori, je dirais tout le modèle.*

