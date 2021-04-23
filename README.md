# TD-de-machine-Learning-Cat-gorisation-musicale_FOUR_Sebastien_LECOMTE_LOUIS
 TD de machine Learning : Catégorisation musicale  Elective : Machine Learning Professeur : Théophile Ravillion Date : Vendredi 23 Avril 2021


# Feuille de route

## Comment procéder ? Pour réaliser cet exercice, il vous faudra :
Récuperer les données
```
In [4]:

import pandas as pd

url_train = "https://raw.githubusercontent.com/RTheophile/td_ml_ynov/main/data/train.csv"
url_test = "https://raw.githubusercontent.com/RTheophile/td_ml_ynov/main/data/test.csv"
df_train = pd.read_csv(url_train, sep=',', decimal='.' )
df_test = pd.read_csv(url_test, sep=',', decimal='.' )
```

## Analyser les données

Identifier la distribution de chaque variable
Données manquantes
Données aberrantes
Données corrélées entre elles

#### Boite à outils :

Décrire le contenu d'un dataframe pandas
```
df.info()
df.describe()
```
Gallerie Seaborn : https://seaborn.pydata.org/examples/index.html#
Histograme : https://seaborn.pydata.org/generated/seaborn.histplot.html
PairPlot : https://seaborn.pydata.org/generated/seaborn.pairplot.html
Corrélogramme : https://seaborn.pydata.org/generated/seaborn.heatmap.html

####Imputation des valeurs manquantes :

KNNImputer : https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html

```
df['chroma_0_mean'] = df['chroma_0_mean'].fillna(df['chroma_0_mean'].mean())
df['chroma_0_mean'] = df['chroma_0_mean'].fillna(df['chroma_0_mean'].median())
df['chroma_0_mean'] = df['chroma_0_mean'].fillna(df['chroma_0_mean'].mode())
```

Quelques fonctions pandas utiles :

Supprimer la ligne i du dataframe df :
```
df.drop(i, axis=0, inplace=True)
```
Supprimer la colonne c du dataframe df :
```
df.drop(i, axis=1, inplace=True)
```
Appliquer la fonction f sur la colonne c du dataframe df :
```
df[c] = df[c].apply(lambda x : f(x))
```

Selectionner les 5 premiers éléments d'un dataframe df :
```
df.head(5)
```

Selectionner les 5 derniers éléments d'un dataframe df :
```
df.tail(5)
```

Selectionner les éléments d'un dataframe qui satisfont une condition :
ex : selectionner toutes les lignes dont le prix est supérieur à 100 :
```
df[df['prix'] > 100]
```

##Normaliser les données

Choisir une méthode de normalisation, comparer les résultats obtenus avec différentes méthodes
Boite à outils

K-NN : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
Robust Scaler : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
StandardScaler : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
MinMaxScaler : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
In [ ]:


Etudier l’opportunité d’une réduction de dimension

    Tester les perfs obtenus pour différentes valeurs
    Visualiser la variance expliquée par chaque axe
    Justifier le nombre d’axes retenus

Boite à outils

Analyse en composante principale : https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
In [ ]:


Créer un échantillon non biaisé de validation

A moins que vous n'utilisiez la k-fold validation (stratifiée ?)
Boite à outil :
```
from sklearn.model_selection import train_test_split
```


Entrainer différents algorithmes de classification
```
model = ...
model.fit(X_train, y_train)
Boite à outils
```

SVM : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
Random Forest : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier
Regression logistique : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
In [ ]:


##Comparer la performance de différents algorithmes

Faire des prédictions et evaluer leur justesse de ces prédictions à l'aide de différents indicateurs :

    Matrice de confusion
    Accuracy
    F-Score

Boite à outils

Un rapport de performance clé en main : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

Liste des différents indicateurs : https://scikit-learn.org/stable/modules/model_evaluation.html

Faire une prédiction avec un model sklearn entrainé : 
```
y_pred_1 = model_1.predict(X_val, y_val)
```

##Optimiser les hyper-paramètres de l’algorithme retenu

Tester différents hyper-paramètres pour tirer au mieux partit de l'algorithme retenu
Boite à outils

GridSearchCV : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html


##Prédire des valeurs pour le jeu de test

Créer un fichier au format .csv contenant vos prédictions. En header le nom des colonnes (music_id et prediction) et pour chacun des morceaux la catégorie prédite.

Vérifier que votre notebook fonctionne avant de le rendre
Boite à outils

exporter un dataFrame pandas au format csv : 
```
df_test.to_csv('data/test.csv' , sep=',', decimal='.')
```


