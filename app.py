# Import packages
from dash import Dash, dcc, html, Input, Output, dash_table, dash
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc


# Incorporate data
df_football_player_21_22 = pd.read_csv('data/2021-2022 Football Player Stats.csv', encoding='ISO-8859-1', delimiter=';') 
df_football_player_22_23 = pd.read_csv('data/2022-2023 Football Player Stats.csv', encoding='ISO-8859-1', delimiter=';')
df_football_team_21_22 = pd.read_csv('data/2021-2022 Football Team Stats.csv', encoding='ISO-8859-1', delimiter=';')
df_football_team_22_23 = pd.read_csv('data/2022-2023 Football Team Stats.csv', encoding='ISO-8859-1', delimiter=';')


# Initialize the app
app = Dash(__name__)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = html.Div([
    html.Div(children='My First App with Data'),
    dash_table.DataTable(data=df_football_player_21_22.to_dict('records'), page_size=10)
])


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
