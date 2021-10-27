# Quelques Theéorie Regression Lineaire
En statistiques, en économétrie et en apprentissage automatique, un modèle de régression linéaire est un modèle de régression qui cherche à établir une relation linéaire entre une variable, dite expliquée, et une ou plusieurs variables, dites explicatives.

# Partie 1 Régression simple


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

Représentation des phénomènes
en réalité en vue de comprendre
le fonctionnement,

Marketing direct en ligne:
construire un modèle pour identifier les
clients les plus susceptibles d’acheter des
produits de leur prochain catalogue

Clients identifiés par le modèle comme
ayant peu de chance d’acheter seront
exclu de la prochaine liste d’envoi.

## La corrélation

Statistique descriptive de la relation entre X et Y: variation
conjointe.

## 1. La covariance

 ###  Dans l’échantillon:

![image 2](images/2.png)

### Estimation pour la population:

![image 3](images/3.png)

### Covariance et nuage de points

![image 4](images/4.png)

## 2. Le coefficient de corrélation linéaire

« de Pearson »

![image 5](images/5.png)

## Modèle

![image 6](images/6.png)

![image 7](images/7.png)

### L’estimation des paramètres

#### a? b?
Méthode d’estimation: les moindres carrés:

![image 8](images/8.png)


```python
dict_ = {'x':[1, 2, 3, 4, 5], 'y': [1, 1, 2, 2, 4]}
df = pd.DataFrame(dict_, dtype=np.float64)

fn_carree = lambda x: x**2
fn_prod = lambda x, y: x*y

df['x_carree'] = np.vectorize(fn_carree)(df.x)
df['y_carree'] = np.vectorize(fn_carree)(df.y)
df['xy'] = np.vectorize(fn_prod)(df.y, df.x)
df
```