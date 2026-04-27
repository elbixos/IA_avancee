<script>
MathJax = {
  tex: {
    inlineMath: {'[+]': [['$', '$']]}
  }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@4/tex-chtml.js"></script>


# L'encodeur des Transformers

Comme son nom l'indique, l'encodeur des transformers a pour objectif d'encoder la séquence d'entrée en une séquence dans laquelle chaque vecteur de la séquence encodée bénéficie d'une [attention](mecanisme_attention.md) relative aux autres vecteurs de la séquence.

En sortie de l'encodeur, la séquence est de taille fixe.

L'encodeur met également en place un certain nombre de mécanismes importants, tels que l'injection d'information de position dans les vecteurs d'entrée.

## Architecture globale de l'encodeur

La figure ci-dessous, extraite de l'article *Attention is all you need* présente cette architecture

![architecture de l'encodeur des Transformers](encoder_transformers.png)







