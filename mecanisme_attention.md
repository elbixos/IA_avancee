<script type="text/javascript" async src="//cdn.bootcss.com/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

## Les Mecanismes d'Attention

On retrouve ces mécanismes dans les Transformers, évidemment, mais ils ont été repris un peu partout dans les réseaux de neurones qui **traitent des séquences**. Voyons donc ceci de plus près.

### Principe de base.

Comme on l'a dit, ces mécanismes d'attention ne sont utiles que pour le traitement de séquences (*eg : texte, séquence temporelle comme des sons, séquence spatiale comme les images*). Modélisons donc ces séquences.

Une séquence d'entrée est une liste des items de la séquence. Notons $$s_s$$ la taille de cette séquence. On notera $$s_i$$, la taille des vecteurs de la séquence.

Une séquence $$S$$ peut donc se mettre sous la forme : $$S = [X_1,...,X_{s_s}]$$, avec $$X_i = [x_1,...,x_{s_i}]$$

On peut aussi la représenter sous une forme matricielle, comme dans l'image ci-dessous :

![Une séquence sous forme matricielle](Images\sequence_initiale.png)

L'idée des mécanismes d'attention est de transformer la séquence pour que chaque vecteur $X_i$ prenne en compte le **contexte** (les autres vecteurs de la séquence).

la sequence d'entrée $$X$$ devient une séquence de sortie $$Y$$ dans laquelle chaque vecteur composant $Y$$ prend en compte certains autres vecteurs de la séquence $$X$$.

Prenons un exemple pour que ce soit plus clair. Imaginons que notre séquence soit composée de mots. Par exemple, la phrase suivante : **le chat rapide mange la souris grise.**

Supposons de plus que chaque mot soit encodé par un vecteur dans un espace sémantique (ou chaque direction possède un sens particulier), 

On peut alors imaginer qu'à l'issue du mécanisme d'attention, chaque mot de la phrase soit modifié comme suit dans $$Y$$ :

- le mot *chat* porte maintenant l'information qu'il est *rapide* (du fait de l'adjectif "rapide"). Il porte aussi peut être l'information qu'il est *masculin*  (du fait de l'article "le"). Enfin, il est peut être *repu*, du fait qu'il a mangé une souris.
- le mot *souris* porte maintenant l'information qu'elle est *grise* (du fait de l'adjectif "grise"), mais peut être aussi qu'elle est *morte*, puisqu'elle est mangée par le chat.

