# Import packages
from dash import Dash, dcc, html, Input, Output, dash_table, dash, callback
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
# app = Dash(__name__)

# app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# badge = dbc.Button(
#     [
#         "Notifications",
#         dbc.Badge("4", color="light", text_color="primary", className="ms-1"),
#     ],
#     color="primary",
# )

# # App layout
# app.layout = html.Div([
#     html.Div(children='My First App with Data'),
#     dash_table.DataTable(data=df_football_player_21_22.to_dict('records'), page_size=10),
#     badge,
# ])



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Clubs Dash', value='tab-1'),
        dcc.Tab(label='Players Dash', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])

@callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Tab content 1')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Tab content 2')
        ])



# Run the app
if __name__ == '__main__':
    app.run(debug=True)
