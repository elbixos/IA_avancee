<script>
MathJax = {
  tex: {
    inlineMath: {'[+]': [['$', '$']]}
  }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@4/tex-chtml.js"></script>

# Apprentissage par Renforcement

## Introduction

L'**Apprentissage par Renforcement**, que l'on appelle aussi *Reinforcement Learning* ou **RL**, est un ensemble de techniques d'apprentissage automatique qui tire parti des résultats d'une **interaction avec un environnement**.

L'objectif est d'associer des situations (les **états**) à des actions à effectuer de façon à maximiser une **récompense** que renvoie l'environnement.

Pour fixer les idées, prenons une partie d'echecs :
- Un état, pour l'apprenant, est la configuration du plateau au moment ou il doit jouer.
- Il va selectionner une action en fonction de cet état.
- il recevra une récompense (des points) en fonction de cette action (positive s'il a pris des pièces adverses ou s'il a gagné, par exemple, et négatives s'il s'est fait prendre des pièces ou qu'il a perdu)

L'objectif du reinforcement learning est qu'un **agent** (un modèle), apprenne par essai-erreur, comment maximiser les récompenses obtenues par ses actions.

L'essentiel de ce qui est présenté ici est tiré d'un livre : *Reinforcement Learning, by Richard S. Sutton and Andrew Barto*, disponible gratuitement un [ici](http://incompleteideas.net/book/RLbook2020.pdf).

*Une petite remarque* : S'il est tentant de considérer l'apprentissage par renforcement comme un apprentissage **non supervisé**, puisqu'on ne cherche pas à aligner les actions de l'agent sur celle d'un professeur. Néanmoins, Sutton considère que non, et qu'**il s'agit d'une classe à part**. *Je ne sais pas trop quoi en penser*.

L'intérêt du RL par rapport aux autres techniques d'IA réside en deux causes :

- lorsqu'on ne peut pas **définir une loss**, mais que c'est l'environnement qui détermine le résultat, bon ou mauvais. (pensez à un agent qui devrait miser en bourse, ou à un agent de conduite autonome, par exemple).
- lorsque la valeur d'une action est impossible à estimer, car seule la **séquence d'action qui détermine le résultat** (pensez aux échecs : un coup peut être bon sur l'instant, mais vous conduire au désastre. Vous pouvez également tout faire bien jusqu'à un tournant fatidique).

A ce titre, le RL est une branche fascinante du machine learning, et ses applications pratiques sont immenses, et déja bien implantées dans le monde industriel.

## Quelques ressources

Les notations du RL sont assez complexes à prendre en main. Il est nécessaire
d'y revenir plusieurs fois, et assez souvent. Pour cela, 
en plus du livre de Sutton, voici quelques liens concernant le RL.

- quelques vidéos en francais : [L4Project video on the subject (actor vs critic)](https://www.youtube.com/watch?v=LtRAgxRb5eQ)
- d'autres excellentes vidéos en anglais [from UCL/Deep Mind]
(https://www.youtube.com/watch?v=TCCjZe0y4Qc&list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm)
- Une explication des [concepts clef](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- une jolie [taxonomie des techniques de RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html#a-taxonomy-of-rl-algorithms)
- une longue description des [détails différenciant les differentes techniques de RL](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)

- Un cours en francais sur le RL, [from CNRS Fidle](https://www.youtube.com/watch?v=8gakDXPHC8c)

## quelques définitions

Pour poser le problème, on considère que notre **agent** évolue dans un **environnement**, par le biais d'**actions**.

A un instant donné, l'agent est considéré comme étant dans un certain **état**,
et percevant de cet environnement une **observation** (observation et état sont assez similaires)

On définit :

- $s_t$ : un **état** à l'instant $t$.
- $A_t$ : la variable aléatoire représentant l'ensemble des **actions** possible, dont $a_t$ est une réalisation

Par exemple, dans un monde déterministe, une action $a_t$ appliquée à un état $s_t$ va conduire à l'état $s_{t+1}$.

L'environnement attribue à l'agent une certaine **récompense** instantanée
lorsque l'agent, dans un état donnée choisit telle action qui le conduit dans tel autre état.

- $r$ : la **récompense** associée au passage d'un état à un autre par le biais d'une action. Il s'agit d'une récompense instantannée.

Pour préciser la notation  temporelle de toute ces grandeurs : A l'instant $t$, l'agent se situe dans un état $s_t$. Il percoit l'observation $o_t$, choisit une action $a_t$. Cette action l'amène dans un état $s_{t+1}$ et il recoit la récompense $r_{t+1}$.

Il s'agit, en Reinforcement Learning, de choisir, à partir d'un état initial, une série d'actions permettant d'obtenir une **récompense cumulée maximale**.

Voyons quelques notions qui vont nous permettre de faire des calculs à partir des notions vues précédemment. On définit ainsi

### Valeur d'un état

- $V(s)$ : la **valeur d'un état**. Quelle récompense cumulée (ou similaire) peut attendre l'agent qui passe par un état $s$. Pour l'évaluer, on aura besoin de la suite :

- $G_t$ : le **retour** : la **récompense cumulée** à partit du temps $t$. Elle s'ecrit comme suit :

$$G_t = R_{t+1} + R_{t+2} + R_{t+3} ...$$

On peut alors écrire $V(s)$ en fonction de $G_t$ :

il s'agit de l'esperance de $G_t$, sachant qu'on est dans l'état $s$
$V(s) = E (G_t \| S_t = s)$

On peut écrire cette équation sous forme récursive :
$V(s) = E (G_t \| S_t = s) = E (R_{t+1} + R_{t+2} + R_{t+3} ... \| S_t = s) = E (R_{t+1} + V(s_{t+1}) \| S_t = s) $

### La Qualité, ou Qvalue d'un couple $s,a$

- $Q(s,a)$ : la qualité d'un couple $s,a$. Si l'agent est dans un état $s$, quelle récompense cumulée (ou similaire) attend-il de l'action $a$.

C'est l'espérance des **retours** connaissant l'état et l'action.
Comme précédement, on écrit :

$$Q(s,a) = E (G_t \| S_t = s, A_t = a)$$

et récursivement

$$Q(s,a) = E (R_{t+1} + V(s_{t+1}) \| S_t = s, A_t = a)$$


### l'historique de l'agent :

on stocke toutes les informations que l'agent a rencontré :

$$H_t = \{o_0, a_0, r_1, o_1, a_1, r_2, ..., r_t, o_t\}$$

(avec $o_t$ l'observation au temps $t$). L'état de l'agent $s_t$ est bati sur $H_t$.

Note pour plus tard : dans le cas ou le monde est completement observable, on peut se contenter de $s_t = o_t$...

## Processus de Décision Markovien (MDP)

**C'est assez peu clair...**

C'est markovien si $p(r,s \| s_t, a_t ) = p(r,s \| H_t, a_t )$
Ca n'a pas l'air d'être le cas si l'environnement est seulement partiellement observable. ou alors, il faut construire un état à partir de l'historique intelligement...

Si c'est markovien, on peut chercher une stratégie optimale.
Mais sinon, on peut se contenter d'une bonne stratégie => on va s'intéresser
à cette stratégie...

### La politique de l'agent

- $\pi$ : une **politique**. une fonction qui, en fonction de l'état, définit l'action à suivre. Dans un monde déterministe, on la note parfois $\mu$ et c'est une fonction $\mu(s) \rightarrow a$. Dans un monde stochastique, c'est une distribution de probabilité, qui à un état associe la probabilité de choisir chacune des actions possibles.

$a = \pi(s)$ ou $\pi(a\|s) = p(a\|s)$

Lions tout ca ensemble et apportons quelques aménagements :

1. La valeur d'un état dépend de la politique d'un agent. On écrira donc
$V_{\pi}(s) = E(G_t \| S_t=s , \pi)$

2. Introduction du **discount factor** $\gamma$ dans le **retour**
$\gamma \in [0,1]$.
Il représente l'intérêt d'une récompense à venir.

Offre les avantages suivant
- de résoudre le cas de séquences de tailles infinies (dans ce cas, sans discount, $G_t \rightarrow \infty$)
- établit un compromis entre récompense immédiate et récompenses futures mais potentielles.

On parle alors de **Discounted Reward**
$G_t = R_{t+1} + \gamma R_{t+2} + \gamma ^2 R_{t+3} ...$

et on garde $V_{\pi}(s) = E(G_t \| S_t=s , \pi)$
sauf que $G_t$ est le discounted reward.

Comme précédement, on peut prendre sa forme récursive :
$G_t = R_{t+1} + \gamma G_{t+1}$

$V_{\pi}(s) = E (G_t \| S_t = s, At \sim \pi(s))$
Avec $a \sim \pi(s)$ qui signifie que $a$ est choisie par la politique $\pi$ dans l'état $s$ (influe sur les probabilités de choisir $a$)

On a donc
$V_{\pi}(s) = E (R_{t+1} + \gamma G_{t+1} \| S_t = s, At \sim \pi(s))$

Ce qui donne **l'équation de Bellmann** :
$V_{\pi}(s) = E (R_{t+1} + \gamma V_{\pi}(s_{t+1}) \| S_t = s, At \sim \pi(s))$

Cette équation est valable aussi pour les valeurs mesurées par la politique optimale, qui donne le meilleur discounted return à chaque état :
$V_{\*}(s) = max_a E (R_{t+1} + \gamma V_{\*}(s_{t+1}) \| S_t = s, A_t = a)$

On peut également écrire l'équation de Bellmann pour la Q_value :
$Q_{\pi}(s,a) = E (R_{t+1} + \gamma Q_{\pi}(s_{t+1},A_{t+1}) \| S_t = s, At = a)$

et sa version pour la Q_value optimale.
$Q_{\*}(s,a) = E (R_{t+1} + \gamma max_a' Q_{\*}(s_{t+1,a}) \| S_t = s, At = a)$


## Apprentissage itératif

L'idée est maintenant que

1. on dispose d'une politique qui détermine un $V_{\pi}(s)$
2. Cette estimation des $V_\{\pi}(s)$ permet de choisir des actions pour aller vers des états plus souhaitables.
3. Ces actions vont sans doute permettre de mettre au point une nouvelle politique.

 L'idée est ainsi de construire itérativement des politiques permettant
 d'approcher $V_{\*}(s)$ (et donc $\pi_{\*}$), ou au moins, des **politiques relativement efficaces**.


## Model

Dans le cas ou l'on dispose d'un modèle, le modèle peut servir à prédire
- l'état suivant : $p(S_{t+1}=s' \| S_t=s, A_t=a)$
- la récompense immédiate suivante $R_{s,a} = E\[R_{t+1} \| S_t = s, A_t =a\]$

Le modèle seul ne donne pas une bonne politique. Il faut prévoir une **plannification** qui elle va optimiser le discounted reward.

## Techniques

Commencons par les techniques adaptées aux **actions discrètes**

## Monte carlo learning :

En RL, quand on parle de Monte Carlo Sampling, on parle d'echantilloner des **épisodes complets**.
un épisode complet est l'ensemble ${(o_i,a_i,r_{i+1})}$, depuis l'état initial, jusqu'à la fin de l'experience (echouée ou réussie, le plus souvent).

On cherche à estimer les valeurs $V(s)$. Ceci peut par exemple être fait à l'aide d'un réseau de neurones. Le même raisonnement s'applique si l'on souhaite estimer $Q(s,a)$.

On initialise les $V(s)$ aléatoirement.
On définit une trajectoire, et on compte la récompense cumulée sur la trajectoire,
pour chaque instant de la trajectoire. Les récompenses antérieures étant sans intérêt pour la valeur de $V(s_t)$, cette récompense cumulée prend la forme suivante :

$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} ... + \gamma^{T-t-1} r_{T}$

On met à jour les valeurs des états $s_t$ avec la formule suivante, ou $\alpha$ est un taux d'apprentissage.
$V(s_t) = V(s_t) + \alpha (G_t - V(s_t))$

### Différence temporelle

Ici, on ne va pas forcément dérouler toute la trajectoire, mais une simple itération
va fournir les informations permettant la mise à jour des estimations (des valeurs ou des qualité).

Pour cela, on va auto estimer la fin de $G_t$. On le comprend en écrivant 
$G_t = r + \gamma G_{t+1}$

Si on suppose que V(s) fournit une estimation à peu près correcte de $G$,
on peut écrire la **différence temporelle**

$$G_t = r_{t+1} + \gamma V(s_{t+1})$$

On parle de **Bootstrapping** : Notre estimation à un instant est évaluée en utilisant cette estimation...

$$V(s_t) = V(s_t) + \alpha (r_{t+1} + \gamma V(s_{t+1}) - V(s_t))$$

La dedans :

- $V(s_t)$ est la prédiction faite par notre estimateur.
- $r_{t+1} + \gamma V(s_{t+1})$ est la valeur cible à atteindre par $V(s_t)$.
- $r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ est l'erreur commise par notre estimateur.
- on note parfois $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$, l'erreur commise par notre estimateur

Ceci fournira la base de SARSA et Qlearning (pas sûr, il me semble avoir vu du Qlearning avec MC)

On le verra plus loin, mais on peut imaginer une différence temporelle à plus d'une itération (disons 3 ou 10...)

### MC vs TD

A noter :
- MC doit attendre la fin de l'épisode pour apprendre
- l'estimation de $V(s)$ est non biaisée avec MC. Elle peut être biaisée avec TD (pas super clair)
- TD Boostrappe. Pas MC.
- **L'estimation de $V$ a une plus grande variance avec MC**. (me semble un peu contre-intuitif)
- MC propage l'information rapidement, TD la propage lentement : Si une récompense inattendue apparait, seul l'état précédent le sait. Mais du coup, les mise à jour de MC sont bruitées. (Me semble lié au précédent)
- MC a semble-t-il besoin d'un $\alpha$ plus faible que TD
- $\alpha$ doit décroitre avec le temps.

### Mixed multi step

au lieu de faire du TD (1-step), ou du MC ($\infty$-step), on peut envisager
des TD à n-step. Mais on peut faire mieux : mixer les deux :

On choisit un parametre $\lambda$, et on ecrit :
$G^\lambda_t = r + \gamma (\lambda  G_{t+1} + (1-\lambda) V(s_{t+1}))$

- si $\lambda = 0$, c'est TD
- si $\lambda = 1$, c'est MC
- pour une valeur $\lambda$, c'est a peu près $1/\lambda$-step TD. Avec $\lambda = 0.1$, on peut considérer qu'on fait à peu près du 10-steps TD. 

Note importante : on peut aussi exprimer $G^\lambda_t$ comme ceci :
$G^{\lambda}_t = \sum_{n=1}^{\infty} (1-\lambda)\lambda^(n-1)G_t(n) $



### Traces...
A revoir.

### SARSA

C'est un exemple de stratégie **on policy**. Sarsa est le premier algo comme cela.

Disons  qu'on dispose d'une stratégie **epsilon-greedy**.

Quand on doit évaluer une qualité : on choisit une action, à l'aide de la policy actuelle. On doit alors estimer la valeur de l'état suivant, que l'on considère être la qualité de l'action suivante selon notre policy actuelle.

Dans un état $s$, on choisit l'action $a$, qui nous donne une récompense $r$ et nous amène dans l'état $s'$, pour lequel on choisit l'action $a'$.
(Sarsa vient de l'enchainement, à chaque itération de $s,a,r,s',a'$ )

$Q(s,a) = Q(s,a) + \alpha (r + \gamma Q(s',a') - Q(s,a))$

### Qlearning

C'est une stratégie off policy, au sens ou pour évaluer la valeur de l'état suivant,
on ne va pas re-appliquer notre politique, mais utiliser $max_{a'} Q(s',a')$.
Visiblement, ça change tout...

$Q(s,a) = Q(s,a) + \alpha (r + \gamma max_{a'} Q(s',a') - Q(s,a))$

### DQN

On utilise simplement un réseau de neurones pour evaluer $q(s,a)$.
Le réseau est décrit comme suit :
- en entrée : $s$
- en sortie : autant de neurones que d'actions possibles $a_i$. Chaque sortie
évalue $q(s,a_i)$

Avantage : Une modification des poids du réseau pour un état modifie le calcul
des sorties pour les autres états. On explore plus vite. Chaque état "diffuse" sur les autres. Cela permet d'estimer des espaces de grande dimension.

Si on écrit le $y$ la sortie du réseau (au sens apprentissage supervisé), c'est :
$y = r + \gamma max_{a'} Q(s',a')$

Si on écrit le $y_pred$ dy réseau (au sens apprentissage supervisé), c'est :
$y = Q(s,a)$

l'erreur du réseau est bien, comme précédemment :
$error = r + \gamma max_{a'} Q(s',a') - Q(s,a)$

Inconvénient : une modification du réseau modifie $Q(s,a)$, mais aussi
$Q(s,a')$ => risque d'instabilité. => on utilise deux réseaux jumeaux

- réseau online : pour calculer $Q(s,a)$
- réseau target : pour calculer $Q(s,a')$

de temps en temps (tous les $C$ steps), on copie online -> target.

### DDQN

### Stochastic reward
Si les récompenses sont probabilistes, on parle de **distributional Q learning**.
(C51 est un exemple)