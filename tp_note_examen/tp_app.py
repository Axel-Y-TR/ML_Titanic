import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np

# Charger le jeu de données
df = pd.read_csv("train.csv").drop(columns=['Unnamed: 0'])
df.dropna(inplace=True)





#Q1 : 
##############################
#TODO  
def deceased_passengers(df):
    
    pclass_1_deceased = df[(df['Pclass'] == 1) & (df['Survived'] == 0)].shape[0]
    pclass_2_deceased = df[(df['Pclass'] == 2) & (df['Survived'] == 0)].shape[0]
    pclass_3_deceased = df[(df['Pclass'] == 3) & (df['Survived'] == 0)].shape[0]
 
    return pclass_1_deceased , pclass_2_deceased , pclass_3_deceased

####################################



#Q2 : 
##############################
#TODO 
 
def survived_passengers(df):
    pclass_1_survived = df[(df['Pclass'] == 1) & (df['Survived'] == 1)].shape[0]
    pclass_2_survived = df[(df['Pclass'] == 2) & (df['Survived'] == 1)].shape[0]
    pclass_3_survived = df[(df['Pclass'] == 3) & (df['Survived'] == 1)].shape[0]

    return pclass_1_survived,  pclass_2_survived , pclass_3_survived
    
####################################



#Q3 : 
##############################
#TODO 

def mean_values(df):
    fare_mean = df['Fare'].mean()
    fare_tax_mean = df['Fare_Tax'].mean()
    luggage_charges_mean = df['Luggage Charges'].mean()
    

    return fare_mean , fare_tax_mean ,luggage_charges_mean

####################################



#Q4 : 
##############################
#TODO 

# Créer une fonction pour le graphique de répartition des passagers par port d'embarquement (Diagramme circulaire)
def create_embarked_pie_chart(df):
    # Compter le nombre de passagers par port d'embarquement
    embarked_counts = df['Embarked'].value_counts()

    # Extraire les noms des ports et les valeurs
    names = embarked_counts.index.tolist()
    values = embarked_counts.values.tolist()

    # Créer le graphique circulaire
    fig = px.pie(names=names, values=values, hole=0.5,
                 title="Répartition des passagers par port d'embarquement")

    return dcc.Graph(figure=fig)


###########################################


#Q5 : 

##################
#TODO

# Créer une fonction pour l'histogramme de survie basé sur la classe
def create_survival_histogram(df):
    fig = px.histogram(df, x="Pclass", color="Survived", barmode='group',
                       category_orders={"Pclass": [1, 2, 3]},
                       title="Histogramme de survie par classe (Pclass)",
                       labels={"Pclass": "Classe", "Survived": "Survie"})

    return dcc.Graph(figure=fig)

 

####################################"



#Q6 :
#########################################"""
#TODO
# Créer une fonction pour le diagramme en boîte basé sur la classe et l'âge

def create_age_box_plot(df):
    df_clean = df.dropna(subset=["Age"])

    # Créer le diagramme en boîte
    fig = px.box(df_clean, x="Pclass", y="Age", color="Pclass",
                 title="Répartition de l'âge par classe",
                 labels={"Pclass": "Classe", "Age": "Âge"})

    return dcc.Graph(figure=fig)

#####################################################""


#Q7 : 
#TODO
#########################################################

def create_fare_line_chart(df):

    fare_mean_by_class = df.groupby("Pclass")["Fare"].mean().reset_index()

    fig = px.line(fare_mean_by_class, x="Pclass", y="Fare",
                  markers=True,
                  title="Tarif moyen par classe",
                  labels={"Pclass": "Classe", "Fare": "Tarif moyen"})

    return dcc.Graph(figure=fig)

###############################################




#Q8 : 
#TODO
#########################################################
# Créer une fonction pour l'histogramme de survie basé sur le sexe
def create_sex_survival_histogram(df):
    fig = px.histogram(df, x="Sex", color="Survived",
                       barmode="group",
                       labels={"Survived": "Survie", "Sex": "Sexe"},
                       title="Répartition des survivants selon le sexe")

    return dcc.Graph(figure=fig)

#########################################################

#Q9 : 
#TODO
#########################################################
# Créer une fonction pour le taux de survie % (Diagramme circulaire)
def create_survival_rate_pie_chart(df):
    survival_counts = df["Survived"].value_counts().sort_index()

    names = ["Non survivants", "Survivants"]
    values = [survival_counts.get(0, 0), survival_counts.get(1, 0)]

    fig = px.pie(
        names=names,
        values=values,
        hole=0.5,
        title="Taux de survie global"
    )

    return dcc.Graph(figure=fig)

#########################################################


#Q10 : 
#TODO
#########################################################

def create_correlation_heatmap(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    correlation_matrix = df[numeric_columns].corr()

    fig = px.imshow(
        correlation_matrix,
        labels=dict(x="Variables", y="Variables", color="Corrélation"),
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        zmin=-1, zmax=1,
        color_continuous_scale="RdBu_r",
        text_auto=True
    )

    fig.update_layout(title="Matrice de Corrélation")

    return dcc.Graph(figure=fig)

#Q11 : 
#TODO
#########################################################
# Fonction pour préparer les données
def prepare_data():
    data = pd.read_csv("train.csv").drop(columns=['Unnamed: 0'])

    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        mean_value = data[col].mean()
        data[col].fillna(mean_value, inplace=True)


    X = data.drop(columns=['Survived'])
    y = data['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


####################################



#Q12 : 
#TODO
#########################################################



# Fonction pour l'entraînement du modèle
def model_training(X_train, y_train, model_type='logistic_regression'):
    if model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)  
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    with open(f'{model_type}_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    return model

##############################################
        

#Q13 : 
#TODO
#########################################################

# Fonction pour tester le modèle
def test_training(X_test, y_test, model_type='logistic_regression'):

    with open(f'{model_type}_model.pkl', 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X_test)

    accuracy = model.score(X_test, y_test)

    return accuracy
################################################"############"


#Q14 : 
#TODO
#########################################################


def predicting_survival(gender, pclass, age, embarked, model_type='logistic_regression'):
    with open(f'{model_type}_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
    embarked_encoding = embarked_mapping.get(embarked, 0)  

    gender_encoding = 0 if gender == 'male' else 1

    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [gender_encoding],
        'Age': [age],
        'SibSp': [0],  
        'Parch': [0],  
        'Fare': [0],
        'Embarked': [embarked_encoding],
        'Fare_Tax': [0],
        'Food Charges': [0],
        'Luggage Charges': [0]
    })

    prediction = model.predict(input_data)[0]  
    return prediction


###########################################


def mean_values_html(df):

    fare_mean , fare_tax_mean ,luggage_charges_mean = mean_values(df)
    return html.Div([
        html.H3("Valeurs moyennes"),
        html.P(f"Valeur moyenne du tarif : {fare_mean}"),
        html.P(f"Taxe moyenne du tarif : {fare_tax_mean}"),
        html.P(f"Valeur moyenne des frais de bagages : {luggage_charges_mean}")
    ], className='three columns')


def survived_passengers_by_class(df):
    pclass_1_survived,  pclass_2_survived , pclass_3_survived =survived_passengers(df)
    return html.Div([
        html.H3("Passagers survivants par classe"),
        html.P(f"Passagers de première classe survivants : {pclass_1_survived}"),
        html.P(f"Passagers de deuxième classe survivants : {pclass_2_survived}"),
        html.P(f"Passagers de troisième classe survivants : {pclass_3_survived}")
    ], className='three columns')

# Créer une fonction pour les passagers décédés par classe
def deceased_passengers_by_class(df):
    pclass_1_deceased , pclass_2_deceased , pclass_3_deceased=deceased_passengers(df)
    return html.Div([
        html.H3("Passagers décédés par classe"),
        html.P(f"Passagers de première classe décédés : {pclass_1_deceased}"),
        html.P(f"Passagers de deuxième classe décédés : {pclass_2_deceased}"),
        html.P(f"Passagers de troisième classe décédés : {pclass_3_deceased}")
    ], className='three columns')







# Créer une fonction pour l'onglet d'entraînement du mod
# Créer l'application Dash
app = dash.Dash(__name__,suppress_callback_exceptions=True)

# Définir la mise en page de l'application
app.layout = html.Div([
    html.H1("🚢 Tableau de bord du Titanic"),
    html.Hr(),
    
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Tableau de bord', value='tab-1'),
        dcc.Tab(label='Entraînement du modèle', value='tab-2'),
        dcc.Tab(label='Détection de survie', value='tab-3'),
    ]),
    
    html.Div(id='tabs-content')
])

# Callback pour mettre à jour le contenu des onglets
@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        # Contenu de l'onglet du tableau de bord
        return html.Div([
            html.Div([
                deceased_passengers_by_class(df),
                survived_passengers_by_class(df),
                mean_values_html(df)
            ], className='row'),
            
            html.Hr(),
            
            html.Div([
                html.Div([
                    create_embarked_pie_chart(df),
                    html.H3("Répartition des passagers par port d'embarquement (Diagramme circulaire)")
                ], className='four columns'),
                
                html.Div([
                    create_survival_histogram(df),
                    html.H3("Histogramme de survie basé sur la classe")
                ], className='four columns'),
                
                html.Div([
                    create_age_box_plot(df),
                    html.H3("Diagramme en boîte basé sur la classe et l'âge")
                ], className='four columns'),
            ], className='row'),
            
            html.Hr(),
            
            html.Div([
                html.Div([
                    create_fare_line_chart(df),
                    html.H3("Diagramme linéaire pour les taxes de tarif, les frais de bagages et les frais de nourriture")
                ], className='four columns'),
                
                html.Div([
                    create_sex_survival_histogram(df),
                    html.H3("Histogramme de survie basé sur le sexe")
                ], className='four columns'),
                
                html.Div([
                    create_survival_rate_pie_chart(df),
                    html.H3("Taux de survie % (Diagramme circulaire)")
                ], className='four columns'),


                html.Div([
                    create_correlation_heatmap(df),
                    html.H3("Matrice de corrélation des variables numériques")
                ], className='four columns'),
                

            ], className='row')
        ])
    
    elif tab == 'tab-2':


        # Contenu de l'onglet d'entraînement du modèle
        return html.Div([
            html.Label("Choisir le type de modèle pour l'entraînement :"),
            dcc.Dropdown(
                id='model-type-dropdown',
                options=[
                    {'label': 'Régression Logistique', 'value': 'logistic_regression'},
                    {'label': 'Forêt Aléatoire', 'value': 'random_forest'}
                ],
                value='logistic_regression'
            ),
            html.Button('Lancer l\'entraînement', id='train-button'),
            html.Div(id='train-message'),

            html.Button('Lancer le test', id='test-button'),
            html.Div(id='test-message')
        ])

    
    elif tab == 'tab-3':
        # Contenu de l'onglet de détection de survie
        return html.Div([
            html.H3("Détection de survie"),
        # Ajouter le menu déroulant pour sélectionner le modèle
            html.Label("Sélectionner le modèle :"),
            dcc.Dropdown(
                id='model-selection-dropdown',
                options=[
                    {'label': 'Régression Logistique', 'value': 'logistic_regression'},
                    {'label': 'Forêt Aléatoire', 'value': 'random_forest'}
                ],
                value='logistic_regression'
            ),
                
            html.Label('Sexe:'),
            dcc.Dropdown(
                id='gender-dropdown',
                options=[
                    {'label': 'Homme', 'value': 'male'},
                    {'label': 'Femme', 'value': 'female'}
                ],
                value='male'
            ),
            
            html.Label('Classe:'),
            dcc.Dropdown(
                id='class-dropdown',
                options=[
                    {'label': '1ère classe', 'value': 1},
                    {'label': '2ème classe', 'value': 2},
                    {'label': '3ème classe', 'value': 3}
                ],
                value=1
            ),
            
            html.Label('Âge:'),
            dcc.Input(id='age-input', type='number', value=30),
            html.Br(),
            html.Label('Port d\'embarquement:'),  # Added label for 'Embarked' dropdown
            dcc.Dropdown(
                id='embarked-dropdown',
                options=[
                    {'label': 'Southampton', 'value': 'S'},
                    {'label': 'Cherbourg', 'value': 'C'},
                    {'label': 'Queenstown', 'value': 'Q'}
                ],
                value='S'
            ),
            
            html.Button('Prédire survie', id='predict-button'),
            html.Div(id='prediction-output')
        ])

X_train, X_test, y_train, y_test=prepare_data()


# Callback pour entraîner le modèle
@app.callback(Output('train-message', 'children'),
              [Input('train-button', 'n_clicks')],
              [State('model-type-dropdown', 'value')])
def train_model(n_clicks, model_type):
    if n_clicks is not None and n_clicks > 0:
        model=model_training(X_train, y_train, model_type=model_type)
        if model is not None : 
            return html.Div(f"Entraînement du modele {model_type} terminé et modèle enregistré.")
        else :
            return html.Div("None")
    


# Callback pour tester le modèle
@app.callback(Output('test-message', 'children'),
              [Input('test-button', 'n_clicks')],
              [State('model-type-dropdown', 'value')])
def test_model(n_clicks, model_type):
    if n_clicks is not None and n_clicks > 0:

        accuracy = test_training(X_test, y_test, model_type=model_type)
        return html.Div(f"Précision du modèle {model_type} : {accuracy}")
    



# Callback pour prédire la survie
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('gender-dropdown', 'value'),
     State('class-dropdown', 'value'),
     State('age-input', 'value'),
     State('embarked-dropdown', 'value'),  # Added State for 'embarked-dropdown'
    State('model-selection-dropdown', 'value')])

def predict_survival(n_clicks, gender, pclass, age, embarked,model_type):
    if n_clicks is not None and n_clicks > 0:
        prediction = predicting_survival(gender, pclass, age, embarked,model_type)
        if prediction == 1:
            return html.Div(f'le modele {model_type } nous dit que la personne aurait survécu.')
        else:
            return html.Div(f'La personne {model_type } nous dit que la personne  aurait péri.')



if __name__ == '__main__':
    app.run(debug=True, host='localhost',port='5050')