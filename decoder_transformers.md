<script>
MathJax = {
  tex: {
    inlineMath: {'[+]': [['$', '$']]}
  }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@4/tex-chtml.js"></script>

# Le décodeur des Transformeurs

Comme dans toute [architecture encodeur/décodeur](encoder_decoder.md), l'objectif est de pouvoir travailler sur un problème **sequence to sequence**,
et de produire **le prochain élement de la séquence de sortie**.

Voyons donc l'architecture du décodeur, présentée ci-dessous :

![architecture du décodeur des Transformers](Images/decoder_transformers.png)

Pour comprendre ce qu'il se passe, imaginons que notre Transformer fasse de la traduction anglais / francais, et, à partir de la phrase "cats eat mouses", il doive au final produire la séquence "les chats mangent des souris".

Plus précisément, au stade qui nous intéresse, à partir du début de traduction "les chats mangent", il doivent produire le mot prochain mot de la séquence de sortie : le mot "des".

Commencons par la partie basse de la figure précédente :

1. Comme dans l'encodeur, la séquence "les chats mangent" est encodée par un réseau dense, appris, de longueur $s \times e$. *J'imagine que la séquence subit un padding pour lui donner une longueur $s$*
2. Cet encodage est aussi complété par un encodage de position, ajouté à l'embedding.

Ce sont ces outputs précédents, encodés, qui vont entrer dans la couche d'attention du décodeur que nous décrivons maintenant :

1. Cet embedding est enrichi par un bloc d'auto-attention multi tête pour que chaque mot de la séquence "les chats mangent" soit enrichi par le contexte des autres mots de cette séquence.
2. Cette version enrichie est encore enrichie par un autre bloc d'attention, cette fois-ci croisée. l'attention est portée à la séquence complète venue de l'encodeur, c'est le context Vector correspondant à la séquence "cats eats mouses". On peut donc imaginer notre séquence comme "les chats mangent", ou chaque mot est enrichi par le contexte de la séquence et par le contexte global "cats eats mouses".
3. Chaque vecteur de cette séquence est également enrichi par l'ajout d'une sortie de réseau feed-forward (*dont l'intérêt ne me saute pas aux yeux*)

Dans le décodeur, ces couches d'attention sont répétées $N$ fois, en série.

Enfin, pour la prédiction finale du prochain token, on utilise un réseau feedforward qui possède $d_{dict}$ sorties, avec $d_{dict}$, le nombre de token possible dans le dictionnaire de langue francaise.





-