<script type="text/javascript" async src="//cdn.bootcss.com/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

## Les Mecanismes d'Attention

On retrouve ces mécanismes dans les Transformers, évidemment, mais ils ont été repris un peu partout dans les réseaux de neurones qui **traitent des séquences**. Voyons donc ceci de plus près.

### Principe de base.

Comme on l'a dit, ces mécanismes d'attention ne sont utiles que pour le traitement de séquences (eg : texte, séquence temporelle comme des sons, séquence spatiale comme les images). Modélisons donc ces séquences.

Une séquence d'entrée est une liste des items de la séquence. Notons $$s_s$$ la taille de cette séquence. On notera $$s_i$$, la taille des vecteurs de la séquence.

Une séquence peut donc se mettre sous la forme : $$ [X_1,...,X_{s_s}]$$, avec $$X_i = [x_1,...,x_{s_i}]$$

On peut aussi la représenter sous une forme matricielle, comme dans l'image ci-dessous :

![Une séquence sous forme matricielle](Images\sequence_initiale.png)


