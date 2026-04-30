<script>
MathJax = {
  tex: {
    inlineMath: {'[+]': [['$', '$']]}
  }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@4/tex-chtml.js"></script>

# Tokenization et Embedding

## Principe général

Dans les LLM, on va injecter du texte pour toutes sortes d'opérations.
Ce texte doit être transformé en **séquences de vecteurs numériques** pour être traité par le modèle. C'est tout l'objectif de la tokenization.

De fait, un **token** est un morceau de texte (disons un mot pour fixer les idées au départ). Le LLM dispose d'un **dictionnaire de token** qui recense tous les tokens possibles. Chaque token à un **identifiant** unique.

Notre texte complet à encoder peut ainsi est remplacé par une succession d'entiers ou chaque entier est l'identifiant d'un token.

De fait, chaque entier (chaque identifiant) est transformé en *one hot vector*.
Notre texte devient donc une séquence de vecteur de taille $s \times d_{dict}$, avec $s$, le nombre de tokens dans le texte et $d_{dict}$, la taille du dictionnaire de token.

*Pour fixer les idées, dans Gpt3, la taille maximale d'une séquence vaut* $s=2048$*, et le dictionnaire de token à une taille* $d_{dict}=50257$*:*

En sortie, un LLM doit prédire un token (le prochain token d'une séquence), Il présente donc $d_{dict}$ neurones de sortie, permettant d'assimiler cette tâche à une classification.

Pour illustrer ceci, voici un exemple de tokenization obtenu avec le [tokenizer d'openAI](https://platform.openai.com/tokenizer) :

![texte tokenisé](Images/tokens_chat_gpt3.png)

Chacune de ces portions colorées est un token, avec son identifiant.
La liste des identifiants du texte proposée est la suivante :

`[2486, 7999, 26433, 731, 549, 2480, 5454, 7936, 1756, 272, 38948]`

## Tokenization

Voyons donc comment cette transformation du texte en tokens est réalisée.
Cela se fait au sein d'un pipeline de tokenization :

1. Normalisation
2. Pré-tokenization
3. Tokenization
4. Post Processing

Voyons un peu plus en détail tout ceci :

1. la normalisation, selon ce que j'en ai lu, consiste à supprimer les accents, mettre tout en minuscules, supprimer les espaces inutiles, pour simplifier le texte à encoder. *Dans la pratique de mes tests, le seul effet que j'ai pu observer est la transformation de "\n" et "\t" en espaces...*
2. la pré-tokenisation consiste à séparer le texte en mots, autour des espaces et de la ponctuation.
3. Le tokenizer : La liste des mot est alors passée au tokenizer, un algorithme pré-entrainé, qui va découper chaque mot en différents token.
4. Le post-processing : il va prendre en charge les espaces multiples, et eventuellement ajouter certains tokens spéciaux pour le LLM

Pour faire des tests plus précis, j'ai du utiliser d'autres tokenizers que ceux d'openAI, non disponibles en python. Les résultats sont donc différents de ceux présentés ci-dessus.

Voici le lien vers les [tests de tokenization](https://colab.research.google.com/drive/1JpRTi_T3KnMCpEzjUKzJN2lfWlTPLwDj?usp=sharing) que j'ai effectués. Je vais en reprendre certaines conclusions ici.

Pour la phrase :

`Le chat porte des \t \n chapeaux chargés d'eau`

le tokenizer produit la séquence :

`['Le', 'chat', 'port', '##e', 'des', 'ch', '##ap', '##eau', '##x', 'ch', '##ar', '##gé', '##s', 'd', "'", 'e', '##au']`

Plus exactement, le tokenizer que j'ai utilisé est celui de BERT, et le post processor ajoute notamment des tokens spéciaux [CLS] et [SEP], qui seront décrit dans la section sur Bert. Voici donc la vraie séquence de tokens générée :

`['[CLS]', 'Le', 'chat', 'port', '##e', 'des', 'ch', '##ap', '##eau', '##x', 'ch', '##ar', '##gé', '##s', 'd', "'", 'e', '##au', '[SEP]']`

qui correspond à cette succession d'identifiants :

`[101, 3180, 13287, 4104, 1162, 3532, 22572, 11478, 8221, 1775, 22572, 1813, 21645, 1116, 173, 112, 174, 3984, 102]`

On peut d'ors et déja noter un point intéressant : le tokenizer ne considère pas de la même façon des successions de caractères de la même façon si elles sont en début de mot, et sinon. Ainsi, le *'eau'* de *'chapeau'* est un token, alors que le mot *'eau'* en fin de phrase est séparé en plusieurs tokens.

Si l'on retourne voir l'exemple de tokens d'openAI en haut, on peut voir, pour la même raison, que les espaces de la phrase sont en fait intégrés dans le token suivant.

### Intérêt de la tokenisation ?

Au plus simple, nous pourrions considérer que chaque caractère est un token.
Il nous suffit de disposer d'un dictionnaire de tous les caractères possibles,
et d'encoder notre texte comme une succession de caractères. C'est simple à réaliser, et le dictionnaire de caractères est très restreint...

Mais, de fait, ce qui porte le sens de nos texte, ce sont les mots, pas les caractères. Donner à un LLM des mots va l'aider à encoder plus simplement le sens des textes. Par exemple, on voit que le mot *'chat'* a un token à lui. Il sera plus facile pour le LLM d'associer ce token aux idées de *mammifère*, *poilu*, *mangeur de souris*, que s'il devait considérer l'ensemble de 4 caractères 'chat' pour lui associer ces idées.

De même, on peut noter dans nos exemples quelques points intéressants :

- le verbe *"porte"* est séparé en "*port*" et *"e"*, ce qui correspond étonnament bien à la racine et à la conjugaison du verbe à la troisième personne du singulier.
- De même, le *"x"* à la fin de *"chapeaux"* porte la marque du pluriel.

Ainsi, la tokenisation permet de découper le texte en petites unités qui portent, idéalement, chacune un peu de sens dans ce texte.

Il faut néanmoins contenir la taille du dictionnaire de tokens, car celle ci
se répercute directement sur la taille du réseau de neurones et sur le nombre de paramètres de celui-ci, donc sur l'espace mémoire qu'il utilise et le temps de calcul qu'il consomme.

### Algorithme de scan de texte

Ici, voyons comment s'est produite la tokenisation de mon texte.

1. le tokenizer dispose d'un dictionnaire de tokens pré-défini
2. le tokenizer scanne le texte à partir de la gauche, et cherche le plus grand token qu'il connait qui matche la chaine du texte. Dans notre exemple, il va trouver *Le*. Il existe un token pour *L*, mais ce n'est pas le plus long. Une fois le token trouvé, il peut recommencer sur la suite de la chaine.

C'est simple et rapide : Si le dictionnaire à une structure de table de hachage, cette tokenisation se fait en $O(n)$ avec $n$, le nombre de charactères du texte.

### Apprentissage de la Tokenisation.

Je ne vais pas m'étendre sur la normalisation, le pre-tokenizer, ni le post-processor, dont le principe semble clair, même s'il recèle des subtilités.

En revanche, le point qui sera développé ici est celui de l'apprentissage du dictionnaire de tokens. Ceci est fait bien en amont du LLM et de façon quasi indépendante.

Il faut tout d'abord disposer d'un ensemble de textes, le **corpus**.
On va appliquer un algorithme, non supervisé, qui va générer un dictionnaire
de tokens pour ce corpus. Il existe plusieurs algorithmes pour cela (Wordpiece, SentencePiece,BPE, BLBPE).

Je vais me concentrer sur le plus simple et qui sert souvent de base aux autres : **BPE**

### BPE : Byte Pairs Encoding

Je vais illustrer le fonctionnement de BPE, avec comme corpus la phrase utilisée précédemment : "Le chat porte des chapeaux chargés d'eau"

Dans **BPE**, la première analyse du corpus permet de générer le premier dictionnaire de tokens, qui contient tous les caractères du corpus.
Dans notre cas, ce serait `[l,e,c,h,a,t,p,o,r,d,s,u,x,g,é,d,']`.

Le corpus est également découpé en mot, à l'aide par normalisation et pré-tokenisation.

Notre encodage du corpus serait donc :

```
|L|e|
|c|h|a|t|
|p|o|r|t|e|
|d|e|s|
|c|h|a|p|e|a|u|x|
|c|h|a|r|g|é|s|
|d|
|'|
|e|a|u|
```

Il s'agit maintenant de déterminer quels tokens ajouter à notre dictionnaire. Ceci sera fait par **fusion** de deux anciens tokens pour en créer un nouveau.
A ce stade, nous allons donc créer un token de deux lettres.

Pour cela, on regarde, dans notre corpus, quelle **paire** de deux tokens existants est **la plus fréquente**. Il s'agit de la paire, *ch*.

1. Celle ci est intégrée à notre dictionnaire de tokens, qui devient
`[l,e,c,h,a,t,p,o,r,d,s,u,x,g,é,d,',ch]`
2. dans notre corpus, on remplace chaque occurence de cette paire par le nouveau token.

Notre encodage du corpus devient donc :

```
|L|e|
|ch|a|t|
|p|o|r|t|e|
|d|e|s|
|ch|a|p|e|a|u|x|
|ch|a|r|g|é|s|
|d|
|'|
|e|a|u|
```

On va réitérer cette opération de recherche de paire dans le nouveau corpus encodé. Ici, les deux paires les plus présentes sont "ch|a" qui se produit 3 fois, et 'a|u' qui se produit 2 fois. A noter que 'e|a' se produit également 2 fois, mais comme on distingue les debuts de mots et les autres, on a en fait une
paire 'e|a' en début de mot et une paire 'e|a' en milieu ou fin.

Après ajout de ces paires, notre dictionnaire devient 

`[l,e,c,h,a,t,p,o,r,d,s,u,x,g,é,d,',ch,cha,au]`

et notre corpus devient :

```
|L|e|
|cha|t|
|p|o|r|t|e|
|d|e|s|
|cha|p|e|au|x|
|cha|r|g|é|s|
|d|
|'|
|e|au|
```

On peut ainsi continuer à ajouter la paire de token la plus fréquente jusqu'à **obtention d'un dictionnaire de taille prédéfinie**. A noter : après chaque ajout d'un token au dictionnaire, il faut ré-encoder le corpus et re-calculer les fréquences. 

### BLBPE : Byte Level Byte Pairs Encoding

**Byte Level BPE** est une petite modification de BPE qui vise à régler un problème majeur : Quand le tokenizer rencontre un caractère qu'il ne connait pas, il encode le token spécial `<UNK>` pour *unknown token*. Ceci arrivait fréquemment lorsque les LLM analysaient du texte contenant des emojis, par exemple.

Pour que tout caractère soit forcément intégré au dictionnaire, il suffit de travailler au niveau des octets (*byte*) du texte a encoder. Ainsi, le dictionnaire de token de base est chacune des 256 valeurs possibles d'un octet.
Puis les paires de token sont réalisées par BPE.

Un caractère en utf-8 est codé sur 1 à 4 octets. Avec ce système, les tokens sont vraisemblablement un peu plus courts (si l'on garde une taille de dictionnaire constant), mais tout caractère rencontré peut être analysé, et prédit.

Enfin, il faut noter qu'étonnament, **les tokenizer des LLM ne sont pas adaptés à une langue spécifique** (en tout cas pour les alphabets occidentaux). C'est le même tokenizer pour un texte en francais ou en anglais (alors que les probabilité d'occurence de motifs sont très différentes).

## Embedding

Ici, je vais revenir sur un point évoqué dans la section concernant [l'encodeur des transformers](encoder_transformers.md) : l'embedding des tokens.

Comme on vient de le voir, un token représente une portion de texte, et encodée comme un entier qui est son identifiant. Pour être traité par un LLM, cet identifiant est transformé en un *one hot vector* dont la dimension est celle du dictionnaire de tokens.

Pour rappel, le dictionnaire de token d'un LLM comme GPT3 a une taille de $d_{dict} ~= 50000$
token. Chaque token est donc encodé comme un one hot vecteur de cette taille.

Prenons un exemple fictif, de tokenizer dans lequel le mot 'chat' serait représenté par un token, et le mot 'chatte' serait représenté par un autre token. Les one hot vector de ces deux tokens sont orthogonaux.

Le rôle de l'embedding est de projeter les vecteurs des tokens dans un espace de dimension $e$. L'objectif est double :

1. réduire le nombre de paramètres du réseau. Si l'on prend $e < d_{dict}$, la taille des matrices utilisées par le réseau est automatiquement réduite, car elle dépend de la taille de l'encodage utilisé.
2. Dans cet espace projeté, les directions des vecteurs ont une connotation sémantique. Ainsi, dans l'espace des embbeding, on peut s'attendre à ce que l'encodage du token "chatte" soit exactement égal, ou très proche, à l'encodage du token "chat" auquel viendrait s'ajouter la direction "féminin".

Cet embbeding est réalisé par une couche dense de neurones standard dont les paramètres sont appris pendant l'entrainement du LLM.

Pour information, dans GPT3, la dimension en sortie de l'embedding est $e = 12288$. Bert utilise, lui un embedding beaucoup plus réduit, avec $e = 768$.

Celui signifie que dans l'espace d'embedding, le réseau dispose de $e$ dimensions pour encoder des significations différentes.

- GPT, qui a vocation à être utilisé sans autre entrainement que son entrainement préalable, a besoin d'une dimension d'embedding beaucoup plus grande, pour encoder toutes sortes de significations, au besoin...
- Bert, lui est pré-entrainé et fine tuné pour des applications précises. Il peut donc travailler avec un espace restreint aux dimensions qui lui seront utiles pour la tâche a exécuter.

