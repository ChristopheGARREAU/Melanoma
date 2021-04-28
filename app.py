from dotenv import load_dotenv

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import pandas as pd
import base64
import json
import datetime
import os
import cv2
import joblib
import numpy as np
from hashlib import blake2s
import pytest

load_dotenv(dotenv_path='./.env')
postgre_host=os.getenv("postgre_host")
postgre_database=os.getenv("postgre_database")
postgre_user=os.getenv("postgre_user")
postgre_password=os.getenv("postgre_password")

print(dcc.__version__)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.config.suppress_callback_exceptions = True

app.title ='Melanoma'

doctor_id = ""

def resize_image(path_picture):
    image = cv2.imread(path_picture)
    height = 106
    width = 106
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def transform_picture_to_predict(picture):
    image = resize_image(picture).astype(np.float32)
    X = np.asarray(image)
    print(X.shape)
    RESHAPED = 33708
    # X_reshaped = X.reshape(X.shape[0], RESHAPED).astype('float32')
    X_reshaped = X.reshape(1, RESHAPED).astype('float32')
    X_reshaped /= 255.
    # PCA_100 = joblib.load('E:/Data/PCA_100.sav')
    PCA_100 = joblib.load('./models/PCA_100.sav')
    X_reshaped_PCA_100 = PCA_100.transform(X_reshaped)
    # model = joblib.load('E:/Data/GradientBoostingClassifier.sav')
    model = joblib.load('./models/GradientBoostingClassifier.sav')
    diagnostic_predicted = model.predict(X_reshaped_PCA_100)
    diagnostic_predicted_proba = model.predict_proba(X_reshaped_PCA_100)

    if diagnostic_predicted[0] == 0 :
        # return "Lésion Bénine", diagnostic_predicted_proba[0][0]
        return "Lésion Bénine", round(diagnostic_predicted_proba[0][0]*100,1)
    else :
        return "Lésion Maline", diagnostic_predicted_proba[0][1]
    # return diagnostic_predicted, diagnostic_predicted_proba

# Test Function
def test_transform_picture_to_predict():
    prediction_test_diag, prediction_test_proba = transform_picture_to_predict('./uploaded_pictures/ISIC_0000008.jpg')
    # assert transform_picture_to_predict('./uploaded_pictures/ISIC_0000008.jpg') == 'Lésion Bénine', 70.2
    assert prediction_test_diag == "Lésion Bénine" and prediction_test_proba == 70.2
    

# SQL Requests

def doctor_patient_list(doc_id):
    conn = psycopg2.connect(host=postgre_host, database=postgre_database, user=postgre_user, password=postgre_password)

    cur = conn.cursor()

    command = (
        """
        SELECT Patient.id, Patient.firstname, Patient.lastname FROM Patient 
        INNER JOIN Docteur ON Docteur.id = Patient.doctor_id
        WHERE doctor_id = %s
        """)

    cur.execute(command, str(doc_id))
    Patients = cur.fetchall()
    cur.close()
    
    # print(Patients)
    Patient_list = []
    Patient_list_id = []

    for i in range(len(Patients)):
        Patient_list.append(' '.join(Patients[i][1:]))
        Patient_list_id.append(Patients[i][0])
    
    return Patient_list, Patient_list_id

def patients_display(doc_id):
    conn = psycopg2.connect(host=postgre_host, database=postgre_database, user=postgre_user, password=postgre_password)

    cur = conn.cursor()

    command = (
        """
        SELECT id, firstname, lastname, gender, birthdate FROM Patient WHERE doctor_id = %s
        """)

    cur.execute(command, str(doc_id))
    Patients = cur.fetchall()
    cur.close()
    
    # Extract the column names
    col_names = list(map(lambda x: x[0], cur.description))

    # Create the dataframe, passing in the list of col_names extracted from the description
    df = pd.DataFrame(Patients, columns=col_names, index=None).set_index('id')
    
    return df

def tumeurs_display(pat_id):
    conn = psycopg2.connect(host=postgre_host, database=postgre_database, user=postgre_user, password=postgre_password)

    cur = conn.cursor()

    command = (
        """
        SELECT id, location, observation FROM Tumeur WHERE patient_id = %s
        """)

    cur.execute(command, str(pat_id))
    Tumeurs = cur.fetchall()
    cur.close()
    
    # Extract the column names
    col_names = list(map(lambda x: x[0], cur.description))

    # Create the dataframe, passing in the list of col_names extracted from the description
    df = pd.DataFrame(Tumeurs, columns=col_names)
    
    return df

def tumor_historic_display(tum_id):
    conn = psycopg2.connect(host=postgre_host, database=postgre_database, user=postgre_user, password=postgre_password)

    cur = conn.cursor()

    command = (
        """
        SELECT Photographie.id, date, picture_path, diagnostic FROM Photographie
        INNER JOIN Tumeur ON Tumeur.id = Photographie.tumeur_id
        INNER JOIN Patient ON Patient.id = Tumeur.patient_id
        WHERE patient_id = %s
        """)

    cur.execute(command, str(tum_id))
    Historic = cur.fetchall()
    cur.close()
    
    # Extract the column names
    col_names = list(map(lambda x: x[0], cur.description))

    # Create the dataframe, passing in the list of col_names extracted from the description
    df = pd.DataFrame(Historic, columns=col_names)
    
    return df

def new_patient(new_firstname, new_lastname, new_gender, new_birthdate, doc_id):
    conn = psycopg2.connect(host=postgre_host, database=postgre_database, user=postgre_user, password=postgre_password)

    cur = conn.cursor()
    command = (
        """
        INSERT INTO Patient (
            firstname,
            lastname,
            gender,
            birthdate,
            doctor_id
            )
        VALUES (%s, %s, %s, %s, (SELECT id from Docteur WHERE id=%s))
        RETURNING id;
        """)

    cur.execute(command, (new_firstname, new_lastname, new_gender, new_birthdate, str(doc_id)))
    conn.commit()

    # get the generated id back
    patient_id = cur.fetchone()[0]

    cur.close()
    
    return print(f"Nouveau Patient créé avec l'id : {patient_id}")

def new_tumor(new_location, new_observation, pat_id):
    conn = psycopg2.connect(host=postgre_host, database=postgre_database, user=postgre_user, password=postgre_password)

    cur = conn.cursor()
    command = (
        """
        INSERT INTO Tumeur (
            location,
            observation,
            patient_id
            )
        VALUES (%s, %s, (SELECT id from Patient WHERE id=%s))
        RETURNING id;
        """)

    #cur.execute(command, record_to_insert)
    cur.execute(command, (new_location, new_observation, str(pat_id)))
    conn.commit()

    # get the generated id back
    tumor_id = cur.fetchone()[0]

    cur.close()
    
    return print(f"Nouvelle Lésion créée avec l'id : {tumor_id}")

def new_picture(new_date, new_picture_path, new_diagnostic, tum_id):
    conn = psycopg2.connect(host=postgre_host, database=postgre_database, user=postgre_user, password=postgre_password)

    cur = conn.cursor()
    command = (
        """
        INSERT INTO Photographie (
            date,
            picture_path,
            diagnostic,
            tumeur_id
        )
        VALUES (%s, %s, %s, (SELECT id from Tumeur WHERE id=%s))
        RETURNING id;
        """)

    #cur.execute(command, record_to_insert)
    cur.execute(command, (new_date, new_picture_path, new_diagnostic, str(tum_id)))
    conn.commit()

    # get the generated id back
    picture_id = cur.fetchone()[0]

    cur.close()
    
    return print(f"Nouvelle Image créée avec l'id : {picture_id}")


# Tables

def wanted_table(df):
    """
    :return: A Div containing table of selected data.
    """
    table = dash_table.DataTable(id='patient_list', 
                                data=df.to_dict('records'),
                                # columns=[{"name": i, "id": i} for i in df.columns],
                                columns=[
                                    {"name": 'Liste des Patients', "id": 'Liste des Patients'}
                                ],
                                style_table={
                                    'height': '40vh',
                                    'padding': 0,
                                    'margin': 0
                                },
                                style_header={
                                                'fontWeight': 'bold',
                                                'backgroundColor': 'rgb(82, 92, 163)',
                                                'color': 'white',
                                                'textAlign': 'center',
                                },
                                style_header_conditional=[
                                    {
                                        'if': {'column_id': 'Liste des Patients'},
                                        'display': 'None'
                                    },
                                    {
                                        'if': {'column_id': 'Patient_id'},
                                        'display': 'None'
                                    },
                                ],
                                style_cell={
                                    'textAlign': 'center',
                                    'whiteSpace': 'no-wrap',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',
                                },
                                style_data_conditional=[
                                                        {
                                                            'if': {'row_index': 'odd'},
                                                            'backgroundColor': 'rgb(225, 225, 240)'
                                                        },
                                                        {
                                                            'if': {'column_id': 'Composant saisi'},
                                                            'textAlign': 'center'
                                                        },
                                                        {
                                                            'if': {"state": "selected"},
                                                            'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                                                            'border': '1px solid blue',
                                                        },
                                ],
                                fixed_rows={'headers': True, 'data': 0},
                                virtualization=True,
                            )
    return table

def wanted_table_2(df):
    """
    :return: A Div containing table of selected data.
    """
    table = dash_table.DataTable(id='patient_tumors_list', 
                                data=df.to_dict('records'),
                                columns=[
                                    {"name": 'Localisation', "id": 'location'},
                                    {"name": 'Observation', "id": 'observation'}
                                ],
                                selected_rows=[],
                                style_table={
                                    'height': '15vh',
                                    'padding': 0,
                                    'margin': 0
                                },
                                style_header={
                                                'fontWeight': 'bold',
                                                'backgroundColor': 'rgb(82, 92, 163)',
                                                'color': 'white',
                                                'textAlign': 'center',
                                                'display': None
                                },
                                style_cell={
                                    'textAlign': 'center',
                                    'whiteSpace': 'no-wrap',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',
                                },
                                style_data_conditional=(
                                                        [{
                                                            'if': {'row_index': 'odd'},
                                                            'backgroundColor': 'rgb(225, 225, 240)'
                                                        },
                                                        {
                                                            'if': {'column_id': 'Composant saisi'},
                                                            'textAlign': 'center'
                                                        }] +
                                                        [{
                                                            'if': {"state": "selected"},
                                                            'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                                                            'border': '1px solid blue'
                                                        } for col in df.columns]
                                ),
                                fixed_rows={'headers': True, 'data': 0},
                                virtualization=True,
                            )
    return table

def wanted_table_3(df):
    """
    :return: A Div containing table of selected data.
    """
    table = dash_table.DataTable(id='tumor_history_list', 
                                data=df.to_dict('records'),
                                columns=[
                                    {"name": 'Date', "id": 'date'},
                                    {"name": 'Chemin de l\'image', "id": 'picture_path'},
                                    {"name": 'Diagnostic', "id": 'diagnostic'}
                                ],
                                selected_rows=[],
                                style_table={
                                    'height': '15vh',
                                    'padding': 0,
                                    'margin': 0
                                },
                                style_header={
                                                'fontWeight': 'bold',
                                                'backgroundColor': 'rgb(82, 92, 163)',
                                                'color': 'white',
                                                'textAlign': 'center',
                                                'display': None
                                },
                                style_cell={
                                    'textAlign': 'center',
                                    'whiteSpace': 'no-wrap',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',
                                },
                                style_data_conditional=[
                                                        {
                                                            'if': {'row_index': 'odd'},
                                                            'backgroundColor': 'rgb(225, 225, 240)'
                                                        },
                                                        {
                                                            'if': {'column_id': 'Composant saisi'},
                                                            'textAlign': 'center'
                                                        },
                                                        {
                                                            'if': {"state": "selected"},
                                                            'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                                                            'border': '1px solid blue',
                                                        },
                                    ],
                                fixed_rows={'headers': True, 'data': 0},
                                virtualization=True,
                            )
    return table

# Layouts

identification_layout = html.Div(
    # I added this id attribute
    id='identification_layout',
    children=[
                # dcc.Link('Go to Page 1', href='/page-1'),
                # html.Br(),
                # dcc.Link('Go to Page 2', href='/page-2'),
                dbc.Row([
                dbc.Col(html.H1("MELANOMA", className="text-center")
                        , className="mb-3 mt-5")
            ]),

            dbc.Row([
                dbc.Col(html.H3(children='Suivi des lésions cutanées de vos patients'
                                        )
                        , className="mb-4 mt-10 text-center")
                ]),

            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        children=[
                            html.H3(children='Identification', className="text-center"),
                            html.Div(
                                [
                                    dbc.Input(id="input_id", placeholder="Identifiant", type="text"),
                                    html.Br(),
                                    dbc.Input(id="input_password", placeholder="Mot de passe", type="password"),
                                ]
                            ),
                                            
                            dbc.Button(html.H4("VALIDER"),
                                id="valid",
                                href="/patient",
                                # color="primary",
                                style={
                                        'background-color': '#525CA3'
                                    },
                                className="mt-3"),

                        ],
                        body=True,
                        # color="dark",
                        style={
                            'border': '3px solid',
                            'border-color': '#525CA3',
                            'height': '30vh'},
                        outline=True,
                    ),
                    width={"size": 6, "offset": 3},
                    className="mb-4"
                ),
            ]),
            ],
    # I added this style attribute
    style={'display': 'block', 'line-height':'0', 'height': '0', 'overflow': 'hidden'}
)

patient_layout = html.Div(
    id='patient_layout',
    children=[
        html.Div(
            id='dashboard_title',
            children=[
                html.Br(),
                dbc.Card(
                    [
                        dbc.CardHeader(html.H2("GESTION DES PATIENTS"), className='card-title', style={'background-color': '#525CA3', 'color': '#ffffff'}),
                    ],
                    style={
                        'border': '3px solid',
                        'border-color': '#525CA3'},
                    # color='primary',
                    outline=True
                ),
                html.Br(),
                
                
            ]
        ),
        html.Div(
            className='row',
            children=[
                #left part
                html.Div(
                    id='patient_view',
                    className = 'col-md-2',
                    children=[
                        html.Div(
                            id='patient_selection',
                            children=[
                                dbc.Card(
                                    [
                                        dbc.CardHeader("LISTE DES PATIENTS", className="card-title", style={'background-color': '#525CA3', 'color': '#ffffff'}),
                                        dbc.CardBody(
                                            [
                                                html.H3("Liste des patients")
                                            ],
                                            id= 'patient_selection_list'
                                        )    
                                    ],
                                    style={
                                        'border': '3px solid',
                                        'border-color': '#525CA3',
                                        'height': '43vh'},
                                    outline=True
                                )             
                            ]
                        ),
                        html.Div(id ='doctor_id_value_save', style={'display': 'none'}),
                        html.Br(),
                        dbc.Card(
                            [
                                dbc.Button(html.H4("NOUVEAU PATIENT"),
                                    id="new_patient",
                                    style={
                                        'background-color': '#525CA3'
                                    }
                                ),
                            ],
                        ),
                        dbc.Modal(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("INFORMATIONS DU NOUVEAU PATIENT", className="card-title", style={'background-color': '#525CA3', 'color': '#ffffff'}),
                                        dbc.ModalBody(
                                            [
                                                dbc.Input(id="input_patient_firstname", placeholder="Prénom", type="text"),
                                                dbc.Input(id="input_patient_lastname", placeholder="Nom", type="text"),
                                                dbc.Input(id="input_patient_sex", placeholder="Sexe", type="text"),
                                                dbc.Input(id="input_patient_birthdate", placeholder="Date de Naissance", type="text"),
                                                html.Br(),
                                                dbc.Button(
                                                    html.H4("VALIDER"),
                                                    id="close-patient-backdrop",
                                                    className="ml-auto",
                                                    style={
                                                    'background-color': '#525CA3'
                                                    }
                                                )
                                            ]
                                        ),
                                    ],
                                    style={
                                        'border': '5px solid',
                                        'border-color': '#525CA3',
                                        'justify-content': 'center'},
                                    outline=True
                                )
                            ],
                            id="patient-modal-backdrop",
                            centered=True,
                            size="sm",
                            backdrop='static'
                        ),
                        html.Br(),
                        dbc.Card(
                            [
                                dbc.Button(html.H4("NOUVELLE LESION"),
                                    id="new_tumor",
                                    style={
                                        'background-color': '#525CA3'
                                    }
                                ),
                            ],
                        ),
                        dbc.Modal(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("INFORMATIONS DE LA NOUVELLE LESION", className="card-title", style={'background-color': '#525CA3', 'color': '#ffffff'}),
                                        dbc.ModalBody(
                                            [
                                                dbc.Input(id="input_tumor_location", placeholder="Localisation", type="text"),
                                                dbc.Input(id="input_tumor_observation", placeholder="Observation", type="text"),
                                                html.Br(),
                                                dbc.Button(
                                                    html.H4("VALIDER"),
                                                    id="close-tumor-backdrop",
                                                    className="ml-auto",
                                                    style={
                                                    'background-color': '#525CA3'
                                                    }
                                                )
                                            ]
                                        ),
                                    ],
                                    style={
                                        'border': '5px solid',
                                        'border-color': '#525CA3',
                                        'justify-content': 'center'},
                                    outline=True
                                )
                            ],
                            id="tumor-modal-backdrop",
                            centered=True,
                            size="sm",
                            backdrop='static'
                        ),
                        html.Br(),
                        dbc.Card(
                            [
                                dbc.Button(html.H4("NOUVELLE IMAGE"),
                                    id="new_picture",
                                    style={
                                        'background-color': '#525CA3'
                                    }
                                ),
                            ],
                        ),
                        dbc.Modal(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("NOUVELLE OBSERVATION DE LA LESION", className="card-title", style={'background-color': '#525CA3', 'color': '#ffffff'}),
                                        dbc.ModalBody(
                                            [
                                                dcc.Upload(
                                                    id = 'uploaded_new_picture',
                                                    children=html.Div(
                                                        [
                                                            html.P("Choisir une image à télécharger")
                                                        ]
                                                    ),
                                                    multiple=False,
                                                    style={
                                                        'borderWidth': '3px',
                                                        'borderStyle': 'dashed',
                                                        'borderRadius': '5px'
                                                    },
                                                ),
                                                html.Br(),
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        html.P("Aucune lésion sélectionnée"),
                                                    ),
                                                    style={
                                                        'border': '3px solid',
                                                        'border-color': '#525CA3',
                                                        'height': '30vh',
                                                        'justify-content': 'center'},
                                                    outline=True,
                                                    id = 'uploaded_tumor_picture'
                                                ),
                                                html.Br(),
                                                html.Div(
                                                    id = 'prediction_display',
                                                    children=[
                                                        html.P("Aucune image sélectionnée")
                                                    ]
                                                ),
                                                # dbc.Input(id="input_picture_date", placeholder="Date", type="text"),
                                                # dbc.Input(id="input_picture_diagnostic", placeholder="Diagnostic", type="text"),
                                                html.Br(),
                                                dbc.Button(
                                                    html.H4("VALIDER"),
                                                    id="close-picture-backdrop",
                                                    className="ml-auto",
                                                    style={
                                                    'background-color': '#525CA3'
                                                    }
                                                )
                                            ]
                                        ),
                                    ],
                                    style={
                                        'border': '5px solid',
                                        'border-color': '#525CA3',
                                        'justify-content': 'center'},
                                    outline=True
                                )
                            ],
                            id="picture-modal-backdrop",
                            centered=True,
                            size="sm",
                            backdrop='static'
                        ),
                    ],
                ),
                
                #right part
                html.Div(
                    id='patient_detail',
                    className='col-md-10',
                    children=[
                        html.Div(
                            id='patient_tumors',
                            children=[
                                dbc.Card(
                                    [
                                        dbc.CardHeader("LESIONS DU PATIENT", className="card-title", style={'background-color': '#525CA3', 'color': '#ffffff'}),
                                        dbc.CardBody(
                                            [
                                                html.H3("Lésions du Patient"),
                                            ],
                                            id= 'patient_tumeurs_list'
                                        )    
                                    ],
                                    style={
                                        'border': '3px solid',
                                        'border-color': '#525CA3',
                                        'height': '20vh',
                                        'justify-content': 'center'},
                                         # color='primary',
                                    outline=True
                                )
                            ]
                        ),
                        html.Div(id ='patient_id_selected_value_save', style={'display': 'none'}),
                        html.Br(),
                        html.Div(
                            id='tumor_historic',
                            children=[
                                dbc.Card(
                                    [
                                        dbc.CardHeader("HISTORIQUE DE LA LESION", className="card-title", style={'background-color': '#525CA3', 'color': '#ffffff'}),
                                        dbc.CardBody(
                                            [
                                                html.P("Aucun historique disponible"),
                                            ],
                                            id= 'tumor_historic_list'
                                        ),
                                    ],
                                    style={
                                        'border': '3px solid',
                                        'border-color': '#525CA3',
                                        'height': '20vh',
                                        'justify-content': 'center'},
                                        # color='primary',
                                    outline=True
                                )
                            ]
                        ),
                        html.Div(id ='tumor_id_selected_value_save', style={'display': 'none'}),
                        html.Br(),
                        html.Div(
                            id='tumor_detail',
                            children=[
                                dbc.Card(
                                    [
                                        dbc.CardHeader("DETAIL DE LA LESION", className="card-title", style={'background-color': '#525CA3', 'color': '#ffffff'}),
                                        dbc.CardBody(
                                            html.Div(
                                                children=[
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    dbc.Card(
                                                                        dbc.CardBody(
                                                                            html.P("Aucune lésion sélectionnée"),
                                                                        ),
                                                                        style={
                                                                            'border': '3px solid',
                                                                            'border-color': '#525CA3',
                                                                            'height': '30vh',
                                                                            'justify-content': 'center'},
                                                                        outline=True,
                                                                        id = 'tumor_picture'
                                                                    ),
                                                                ],
                                                                width=6
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    dbc.Card(
                                                                        dbc.CardBody(
                                                                            [
                                                                                dbc.Row(
                                                                                    [
                                                                                        dbc.Col(
                                                                                            [
                                                                                                html.P("Aucune lésion sélectionnée", className="card-text"),
                                                                                            ]
                                                                                        )
                                                                                    ]
                                                                                )
                                                                            ]
                                                                        ),
                                                                        className="card-text",
                                                                        style={
                                                                            'border': '3px solid',
                                                                            'border-color': '#525CA3',
                                                                            'height': '30vh',
                                                                            'justify-content': 'center'},
                                                                        outline=True,
                                                                        id= 'tumor_diagnostic'
                                                                    )
                                                                ],
                                                                width=6
                                                            ),
                                                        ],
                                                        align='center',
                                                        no_gutters=True
                                                    )
                                                ]
                                            )
                                        )    
                                    ],
                                    style={
                                        'border': '3px solid',
                                        'border-color': '#525CA3',
                                        'height': '35vh'},
                                        # color='primary',
                                    outline=True
                                )
                            ]
                        ),
                        html.Br()
                    ]
                )
            ]
        )
    ]
)





page_1_layout = html.Div(
    # I added this id attribute
    id='page_1_layout',
    children=[
        html.H1('Page 1'),
        dcc.Dropdown(
            id='page-1-dropdown',
            options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
            value='LA'
        ),
        html.Div(id='page-1-content'),
        html.Br(),
        dcc.Link('Go to Page 2', href='/page-2'),
        html.Br(),
        dcc.Link('Go back to home', href='/'),
    ],
    # I added this style attribute
    style={'display': 'block', 'line-height': '0', 'height': '0', 'overflow': 'hidden'}

)

page_2_layout = html.Div(
    # I added this id attribute
    id='page_2_layout',
    children=[
        html.H1('Page 2'),
        html.Div(id='page-2-content'),
        html.Br(),
        dcc.Link('Go to Page 1', href='/page-1'),
        html.Br(),
        dcc.Link('Go back to home', href='/'),
    ],
    # I added this style attribute
    style={'display': 'block', 'line-height': '0', 'height': '0', 'overflow': 'hidden'}
)

# Layout to keep in last position
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content',
             # I added this children attribute
             children=[identification_layout, patient_layout, page_2_layout]
             )
])


# Callbacks

# Update the index
@app.callback(
    [dash.dependencies.Output(page, 'style') for page in ['identification_layout', 'patient_layout', 'page_2_layout']],
    # I turned the output into a list of pages
    [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    return_value = [{'display': 'block', 'line-height': '0', 'height': '0', 'overflow': 'hidden'} for _ in range(3)]

    if pathname == '/page-1':
        return_value[1] = {'height': 'auto', 'display': 'inline-block'}
        return return_value
    elif pathname == '/page-2':
        return_value[2] = {'height': 'auto', 'display': 'inline-block'}
        return return_value
    elif pathname == '/patient':
        return_value[1] = {'height': 'auto'}
        return return_value
    else:
        return_value[0] = {'height': 'auto'}
        return return_value

# 
@app.callback(dash.dependencies.Output('page-1-content', 'children'),
              [dash.dependencies.Input('page-1-dropdown', 'value')])
def page_1_dropdown(value):
    return 'You have selected "{}"'.format(value)

# 
@app.callback(Output('page-2-content', 'children'),
              [Input('page-1-dropdown', 'value')])
def page_2(value):
    return 'You selected "{}"'.format(value)

# Patients List
@app.callback(
    # Output('patient_selection', 'children'),
    [Output('patient_selection_list', 'children'),
    Output('doctor_id_value_save', 'children')],
    [Input('valid', 'n_clicks'),
    State('input_id', 'value'),
    State('input_password', 'value')]
)
def doc_id_get(n, doc_id, doc_pass):

    # doc_pass = blake2s(b'Mad33+').hexdigest()
    if doc_pass != None:
        doc_pass = blake2s(doc_pass.encode(encoding="utf-8")).hexdigest()

    conn = psycopg2.connect(host=postgre_host, database=postgre_database, user=postgre_user, password=postgre_password)

    cur = conn.cursor()

    command = (
        """
        SELECT id FROM Docteur WHERE identification_id = %s AND password = %s
        """
        )

    cur.execute(command, (doc_id, doc_pass))
    toubib = cur.fetchall()
    cur.close()
    
    
    if len(toubib) > 0:
        doctor_id = toubib[0][0]
        # print(doctor_id)
        Patient_list, Patient_list_id = doctor_patient_list(toubib[0][0])
        df = pd.DataFrame(list(zip(Patient_list_id, Patient_list)), columns=['Patient_id','Liste des Patients'])
        table = wanted_table(df)
        return table, json.dumps(doctor_id)
    else:
        return [], ""

# Tumors List
@app.callback(
    [Output('patient_tumeurs_list', 'children'),
    Output('patient_id_selected_value_save', 'children')],
    [Input('patient_list', 'active_cell'),
    State('patient_list', 'data')]
)
def getActiveCell(active_cell, data):
    if active_cell:
        col = active_cell['column_id']
        row = active_cell['row']
        # cellData = data[row][col]

        patient_id_selected = data[row]['Patient_id']

        df = tumeurs_display(patient_id_selected)
        
        if df.empty != True :
            tumeurs = wanted_table_2(df)

            return tumeurs, json.dumps(patient_id_selected)
            
        else :
            return html.P('Aucune lésion associée au patient'), json.dumps(patient_id_selected)    
                
    return html.P('Aucun Patient sélectionné'), ""  

# Tumor Historic        
@app.callback(
    [Output('tumor_historic', 'children'),
    Output('tumor_id_selected_value_save', 'children')],
    [Input('patient_tumors_list', 'active_cell'),
    State('patient_tumors_list', 'data')]
)
def getActiveCell2(active_cell, data):
    if active_cell:
        col = active_cell['column_id']
        row = active_cell['row']
        # cellData = data[row][col]

        tumor_id_selected = data[row]['id']
        print(tumor_id_selected)

        df = tumor_historic_display(tumor_id_selected)
        
        if df.empty != True :
        
            historique = [
                dbc.Card(
                    [
                        dbc.CardHeader("HISTORIQUE DE LA LESION", className="card-title", style={'background-color': '#525CA3', 'color': '#ffffff'}),
                        dbc.CardBody(
                            wanted_table_3(df),
                        )    
                    ],
                    style={
                        'border': '3px solid',
                        'border-color': '#525CA3',
                        'height': '20vh',
                        'justify-content': 'center'},
                    # color='primary',
                    outline=True
                ),
            ]
            return historique, json.dumps(tumor_id_selected)
            
        else :
            return [dbc.Card(
                    [
                        dbc.CardHeader("HISTORIQUE DE LA LESION", className="card-title", style={'background-color': '#525CA3', 'color': '#ffffff'}),
                        dbc.CardBody(
                            html.P('Aucun historique disponible')
                        )    
                    ],
                    style={
                        'border': '3px solid',
                        'border-color': '#525CA3',
                        'height': '20vh',
                        'justify-content': 'center'},
                    # color='primary',
                    outline=True
                ),
            ], json.dumps(tumor_id_selected)
                
    return [dbc.Card(
                    [
                        dbc.CardHeader("HISTORIQUE DE LA LESION", className="card-title", style={'background-color': '#525CA3', 'color': '#ffffff'}),
                        dbc.CardBody(
                            html.P('Aucun historique disponible')
                        )    
                    ],
                    style={
                        'border': '3px solid',
                        'border-color': '#525CA3',
                        'height': '20vh',
                        'justify-content': 'center'},
                    # color='primary',
                    outline=True
                ),
    ], ""

# Tumor Picture Display
@app.callback(
    [Output('tumor_picture', 'children'),
    Output('tumor_diagnostic', 'children')],
    [Input('tumor_history_list', 'active_cell'),
    State('tumor_history_list', 'data')]
)
def getActiveCell3(active_cell, data):
    if active_cell:
        col = active_cell['column_id']
        row = active_cell['row']
        # cellData = data[row][col]

        pic_path = data[row]['picture_path']
        pic_diag = data[row]['diagnostic']

        if pic_path != "":
            encoded_image = base64.b64encode(open(pic_path, 'rb').read())

            return [dbc.CardImg(src='data:image/jpg;base64,{}'.format(encoded_image.decode()), className='align-self-center', style={'height': '100%', 'width': 'auto'}),
                    html.H2(pic_diag)]

        else:
            return [html.P("Aucune image sélectionnée"),
                    html.P("Aucun diagnostic associé")]

    return [html.P("Aucune image sélectionnée"),
            html.P("Aucun diagnostic associé")]

# New Patient Insertion
@app.callback(
    Output("patient-modal-backdrop", "is_open"),
    [
        Input("new_patient", "n_clicks"),
        Input("close-patient-backdrop", "n_clicks"),
        Input('doctor_id_value_save', 'children'),
    ],
    [
        State("patient-modal-backdrop", "is_open"),
        State("input_patient_firstname", "value"),
        State("input_patient_lastname", "value"),
        State("input_patient_sex", "value"),
        State("input_patient_birthdate", "value"),
    ],
)
def new_patient_modal(n1, n2, doctor_id, is_open, firstname, lastname, sex, date):
    if n1 or n2:
        if n2:
            doctor_id = json.loads(doctor_id)
            print(f"Le patient saisi est : {firstname} / {lastname} / {sex} / {date}")
            new_patient(firstname, lastname, sex, date, int(doctor_id))
        return not is_open
    return is_open

# New Tumor Insertion
@app.callback(
    Output("tumor-modal-backdrop", "is_open"),
    [
        Input("new_tumor", "n_clicks"),
        Input("close-tumor-backdrop", "n_clicks"),
        Input('patient_id_selected_value_save', 'children'),
    ],
    [
        State("tumor-modal-backdrop", "is_open"),
        State("input_tumor_location", "value"),
        State("input_tumor_observation", "value"),
    ],
)
def new_tumor_modal(n1, n2, patient_id_selected, is_open, location, observation):
    if n1 or n2:
        if n2:
            patient_id = json.loads(patient_id_selected)
            print(f"La tumeur saisie est : {location} / {observation}")
            new_tumor(location, observation, patient_id)
        return not is_open
    return is_open

# New Picture Insertion
@app.callback(
    [
        Output('uploaded_tumor_picture', "children"),
        Output('prediction_display', "children"),
        Output("picture-modal-backdrop", "is_open"),
    ],
    [
        Input("new_picture", "n_clicks"),
        Input('uploaded_new_picture', "contents"),
        Input("close-picture-backdrop", "n_clicks"),
        Input('tumor_id_selected_value_save', 'children'),
    ],
    [
        State("picture-modal-backdrop", "is_open"),
        State('uploaded_new_picture', "filename"),
        State('uploaded_new_picture', "last_modified"),
        # State("input_picture_date", "value"),
        # State("input_picture_diagnostic", "value"),
    ],
)
# def new_picture_modal(n1, n2, is_open, date, diagnostic):
# def new_picture_modal(n1, uploaded_picture, n2, is_open, uploaded_picture_filename, uploaded_picture_date, date, diagnostic):
def new_picture_modal(n1, uploaded_picture, n2, tumor_id_selected, is_open, uploaded_picture_filename, uploaded_picture_date):    
    
    # if n1 or n2:
    #     if n2:
    #         print(f"La photo saisie est : {date} / {diagnostic}")
    #     return not is_open
    # return is_open

    if n1 or n2:
        if uploaded_picture is not None:
            if n2:
                today_date = datetime.date.today()
                print(f"La date est : {datetime.date.today()}")

                data = uploaded_picture.encode("utf8").split(b";base64,")[1]
                UPLOAD_DIRECTORY = "./uploaded_pictures"
                with open(os.path.join(UPLOAD_DIRECTORY, uploaded_picture_filename), "wb") as fp:
                    fp.write(base64.decodebytes(data))

                path_picture_uploaded = UPLOAD_DIRECTORY + "/" + str(uploaded_picture_filename)

                diagnostic_predicted, diagnostic_predicted_proba = transform_picture_to_predict(UPLOAD_DIRECTORY + "/" + str(uploaded_picture_filename))

                new_picture(today_date, path_picture_uploaded, diagnostic_predicted, tumor_id_selected)

                return html.Img(src=uploaded_picture), html.P("Aucune image sélectionnée"), not is_open
            else:
                data = uploaded_picture.encode("utf8").split(b";base64,")[1]
                UPLOAD_DIRECTORY = "./uploaded_pictures"
                with open(os.path.join(UPLOAD_DIRECTORY, uploaded_picture_filename), "wb") as fp:
                    fp.write(base64.decodebytes(data))

                print(UPLOAD_DIRECTORY + "/" + str(uploaded_picture_filename))
                diagnostic_predicted, diagnostic_predicted_proba = transform_picture_to_predict(UPLOAD_DIRECTORY + "/" + str(uploaded_picture_filename))

                return html.Img(src=uploaded_picture), html.P([diagnostic_predicted, " à ", str(round(diagnostic_predicted_proba*100,1)), "%"]), is_open
        else:
            if n2:
                print(f"La photo saisie est : {date} / {diagnostic}")
            return html.P(""), html.P("Aucune image sélectionnée"), not is_open
    return html.P(""), html.P("Aucune image sélectionnée"), is_open


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True, port=8051)
    