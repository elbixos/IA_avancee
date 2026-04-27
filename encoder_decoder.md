<script>
MathJax = {
  tex: {
    inlineMath: {'[+]': [['$', '$']]}
  }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@4/tex-chtml.js"></script>

# Architecture Encoder/Decoder

Avant de présenter l'architecture Encodeur/Décodeur, il me semble nécessaire de vous présenter quelques types d'applications différents, qui permettront de comprendre pourquoi cette architecture a été si importante dans le traitement de données séquentielles.

## Typologie des applications

Dans le monde du machine learning, et en particulier pour les réseaux récurrents, on distinguait historiquement différents types d'applications,
en fonction du nombre de vecteurs en entrée et en sortie du réseau.
Quand je parle de nombre de vecteurs, je parle de la distinction entre données non séquentielles et données séquentielles. 

Voici la liste de ces types d'applications :

- One to One
- Many to One
- One to Many
- Many to Many.

Par exemple, un problème de classification, ou de régression, consiste toujours, pour chaque vecteur d'entrée (One), à fournir un vecteur de sortie (One). C'est un problème **One to One**.

Un problème d'annotation d'image consiste, pour chaque une image d'entrée (One), à fournir une liste de labels décrivants l'image (Many). C'est un problème **One to Many**

Un problème de classification de commentaires textuels, consiste à prendre une **séquence** en entrée (Many), donc une liste de vecteurs, pour produire une classe ou une distribution de probabilité de classe, donc un vecteur ou un scalaire (One) en sortie. C'est un problème **Many to One**

Enfin, un problème de traduction de texte prend en entrée une séquence (Many) et produit en sortie une séquence (Many). C'est un problème **Many to Many**. Dans ce cas, il était parfois utile de faire la distinction entre modèle capables de travailler sur des séquences de tailles différente en entrée et en sortie... 

L'architecture Encodeur/Décodeur a été inventée pour traiter les problèmes de type **Many to Many**, appelés aussi **sequence to sequence**.

Pour tout ce qui suit, nous prendrons l'exemple d'une application de traduction anglais francais, avec comme entrée la phrase "*cats eat mouses*", pour produire en sortie "*les chats mangent avec des souris*".

## Rôle de l'encodeur

L'encodeur a pour objectif de fournir **une représentation de la séquence complète**. Il va pouvoir regarder l'ensemble de la séquence pour en fournir une description numérique appelée **Context Vector**. 

Cette représentation peut être un vecteur unique, dans un espace dit **latent**. Dans cet espace latent, notre phrase est représentée par un point unique, signifiant que les chats mangent des souris...

Il arrive également que ce Context Vector soit une séquence de vecteurs.
Dans ce cas, chaque mot de la phrase d'entrée (chaque token, en fait), va être travaillé par l'encodeur en fonction du contexte de la phrase complète pour représenter un concept plus précis. Par exemple, les souris évoquées par cette phrases sont a priori des mammifères et non pas du matériel informatique.

## Rôle du décodeur

l'objectif du décodeur est simplement de **prédire le mot suivant de la séquence de sortie**.

Pour cela, il dispose de plusieurs informations :

- le context vector (venant de l'encodeur)
- les mots précédents de la séquence de sortie.

Ainsi, si son objectif est de produire le prochain mot de la traduction "le chat mange" correspondant à "cats eat mouses", il recevra en entrée le context vector correspondant à la phrase anglaise complète (qui peut être un vecteur ou une séquence), et la séquence "le chat mange". Il devra produire alors produire le mot "des".

En phase d'inférence, le décodeur va travailler mot après mot pour générer tout la séquence de sortie, jusqu'à produire un token "fin de séquence".


