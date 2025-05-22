# ML_Titanic

## installation 

Création de l'environnement virtuel et installation des dépendances

Assurez-vous d'avoir Python installé. Vous pouvez vérifier en exécutant 

    python --version 

Ouvrez un terminal.

    python -m venv venv

Activez l'environnement virtuel. Selon votre système d'exploitation, utilisez la commande appropriée ci-dessous :

    python -m venv venv

Activez l'environnement virtuel. Selon votre système d'exploitation, utilisez la commande appropriée ci-dessous :

    source venv/bin/activate


Installez les dépendances à partir du fichier requirements.txt. Exécutez la commande suivante :

    pip install -r requirements.txt

Lancer le code

    python tp_ap.py



---------------------------
Pour chaque fonction à écrire dans votre code, vous trouverez un commentaire identifié par la question, suivie de l'annotation @TODO ET Q1 ,Q2 ,Q3, Q4 ... pour indiquer la tâche à accomplir. Assurez-vous de remplacer ces annotations @TODO par le code approprié pour chaque fonction.
--------------------------------
## Question 1 

    Écrivez une fonction nommée deceased_passengers(df) qui prend un DataFrame df en entrée contenant des données sur les passagers du Titanic. Cette fonction doit retourner un tuple contenant trois éléments :

        Le nombre de passagers décédés dans la classe 1 (Pclass == 1).
        Le nombre de passagers décédés dans la classe 2 (Pclass == 2).
        Le nombre de passagers décédés dans la classe 3 (Pclass == 3).


## Question 2


    Écrivez une fonction nommée survived_passengers(df) qui prend un DataFrame df en entrée contenant des données sur les passagers du Titanic. La fonction doit retourner un tuple contenant trois éléments :

        Le nombre de passagers ayant survécu dans la classe 1 (Pclass == 1).
        Le nombre de passagers ayant survécu dans la classe 2 (Pclass == 2).
        Le nombre de passagers ayant survécu dans la classe 3 (Pclass == 3).


## Question 3 :

    Écrivez une fonction nommée mean_values(df) qui prend un DataFrame df en entrée contenant des données sur les passagers du Titanic. La fonction doit retourner un tuple contenant trois éléments :

        La moyenne des tarifs des billets (Fare).
        La moyenne des taxes sur les tarifs (Fare_Tax).
        La moyenne des frais de bagages (Luggage Charges).

## Question 4 :

    Écrivez en utilisant la bibliothèque Plotly Express pour créer un diagramme circulaire représentant la répartition des passagers du Titanic en fonction du port d'embarquement (Embarked). Assurez-vous que le diagramme a un trou central pour améliorer sa lisibilité.

    Consigne :

        Utilisez le DataFrame df contenant les données sur les passagers du Titanic.
        La ligne de code doit utiliser la fonction px.pie() de Plotly Express avec les paramètres appropriés pour créer le diagramme circulaire avec un trou central (hole=0.5).


## Question 5 :

    Écrivez une fonction  create_survival_histogram(df) qui prend en entrée un DataFrame df contenant les données des passagers du Titanic. La fonction doit retourner un histogramme représentant la répartition des passagers en fonction de leur survie, en distinguant les différentes classes avec des couleurs différentes et en affichant les données groupées par classe.


## Question 6 :


    Écrivez une fonction  nommée create_age_box_plot(df) qui prend un DataFrame df en entrée contenant des données sur les passagers du Titanic. Cette fonction doit retourner un diagramme en boîte représentant la distribution de l'âge des passagers en fonction de leur classe (Pclass).


## Question 7:

    Écrivez une fonction  nommée create_fare_line_chart(df) qui prend un DataFrame df en entrée contenant des données sur les passagers du Titanic. Cette fonction doit retourner un graphique linéaire représentant l'évolution des frais (Fare) ainsi que des composantes de ces frais, à savoir la taxe sur le tarif (Fare_Tax), les frais de bagages (Luggage Charges) et les frais de nourriture (Food Charges).


## Question 8:


    Ecrivez une fonction  nommée create_sex_survival_histogram(df) df contenant des données sur les passagers du Titanic. La fonction doit retourner un histogramme représentant la répartition des passagers en fonction de leur survie, en distinguant les sexes masculin et féminin.





## Question 9:


    Écrivez une fonction  nommée create_survival_rate_pie_chart(df) qui prend un DataFrame df en entrée contenant des données sur les passagers du Titanic. Cette fonction doit retourner un diagramme circulaire représentant le taux de survie des passagers.

## Question 10 :


    Écrivez une fonction  nommée create_correlation_heatmap(df) qui prend un DataFrame df en entrée contenant des données numériques. La fonction doit créer une carte thermique de corrélation représentant les corrélations entre les différentes variables numériques du DataFrame.

## Question  11 :

	Écrivez une fonction  nommée prepare_data() qui ne prend aucun argument en entrée explicite.

	Cette fonction doit charger un jeu de données à partir d'un fichier CSV nommé.

	Supprimez les valeurs manquantes du jeu de données.

	Remplacez les valeurs catégorielles par des valeurs numériques.

	Séparez les données en ensembles d'entraînement et de test en utilisant la fonction train_test_split() avec un rapport de test de 20%.


## Question 12 :

        Écrivez une fonction  nommée model_training(X_train, y_train, model_type='logistic_regression') prenant trois arguments en entrée : X_train, y_train et model_type='logistic_regression'.

        Dans la fonction, utilisez une structure conditionnelle pour créer un modèle en fonction de la valeur de model_type. Si model_type est égal à 'logistic_regression', créez un modèle de régression logistique (LogisticRegression()), sinon si model_type est égal à 'random_forest', créez un modèle de forêt aléatoire (RandomForestClassifier()).

        Si model_type n'est ni 'logistic_regression' ni 'random_forest', lancer une ValueError avec le message "Type de modèle invalide. Veuillez choisir entre 'logistic_regression' et 'random_forest'".

        Entraînez le modèle avec les données d'entraînement (X_train, y_train).

        Enregistrez le modèle entraîné dans un fichier avec une extension ".pkl" correspondant au type de modèle utilisé (par exemple, "logistic_regression_model.pkl" pour la régression logistique).

        La fonction doit retourner le modèle entraîné.


## Question 13 :

        Écrivez une fonction  nommée test_training(X_test, y_test, model_type='logistic_regression') prenant trois arguments en entrée : X_test, y_test et model_type='logistic_regression'.

        Ouvrez le fichier contenant le modèle entraîné en fonction du model_type spécifié en mode lecture binaire ('rb').

        Chargez le modèle à partir du fichier.

        Utilisez le modèle pour prédire les valeurs cibles (y_pred) en utilisant les données de test (X_test).

        Calculez la précision du modèle en comparant les valeurs prédites (y_pred) avec les vraies valeurs cibles (y_test) en utilisant la fonction accuracy_score().

        Retournez la précision calculée.

## Question  14:

        Écrivez une fonction  nommée predicting_survival(gender, pclass, age, embarked,model_type='logistic_regression') prenant cinq arguments en entrée : gender, pclass, age, embarked et model_type='logistic_regression'.

        Ouvrez le fichier contenant le modèle entraîné en fonction du model_type spécifié en mode lecture binaire ('rb').

        Chargez le modèle à partir du fichier.

        Encodez la valeur de embarked en utilisant le dictionnaire {'S': 0, 'C': 1, 'Q': 2}. Si la valeur de embarked n'est pas trouvée dans le dictionnaire, utilisez une valeur par défaut de 0.

        Préparez les données de prédiction en créant un DataFrame contenant les valeurs de pclass, gender, age, embarked_encoding ainsi que des valeurs de placeholders pour les caractéristiques manquantes (SibSp, Parch, Fare, Fare_Tax, Food Charges, Luggage Charges).

        Utilisez le modèle pour prédire la survie en utilisant les données de prédiction.

        Retournez la prédiction.
