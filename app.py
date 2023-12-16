import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
from datetime import datetime as dt
import pathlib

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.metrics import silhouette_score

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Football stats Analytics Dashboard"

server = app.server
app.config.suppress_callback_exceptions = True

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

"""
numerical processing
"""
def get_numeric_df(df, key):
    df_numeric = None
    if 'Team' in key:
        squads = df.get('Squad')
        df.drop(['Rk', 'Squad', 'Country', 'Top Team Scorer', 'Goalkeeper'], axis=1, inplace=True)

        # Handle any missing values
        df.fillna(df.mean(), inplace=True)
        df['GD'] = df['GF'] - df['GA']  # Recalculating Goal Difference
        df['Pts/MP'] = df['Pts'] / df['MP']  # Recalculating Points per Match

        # Feature Engineering
        # add or modify features based on domain knowledge
        df['WinRate'] = df['W'] / df['MP']  # Example: Win rate
        df['GoalDifferencePerMatch'] = df['GD'] / df['MP']  # Goal difference per match

        # Select Features for Clustering
        features = ['MP', 'WinRate', 'D', 'L', 'GF', 'GA', 'GoalDifferencePerMatch', 'Pts/MP', 'xG', 'xGA', 'xGD', 'xGD/90', 'Attendance']
        df_numeric = df[features]

        return df_numeric, squads
    
    elif 'Player' in key:
        players = df['Player']
        df.drop(['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Born'], axis=1, inplace=True)

        # Handle any missing values
        df.fillna(df.mean(), inplace=True)

        # Select Features for Clustering
        # Add or modify features based on domain knowledge
        features = ['MP', 'Starts', 'Goals', 'Shots', 'SoT%', 'G/Sh', 'G/SoT', 'PasTotCmp%', 'PasTotDist', 'Assists', 'SCA', 'GCA', 'Tkl', 'Touches', 'CrdY', 'CrdR', 'AerWon%']
        df_numeric = df[features]

        return df_numeric, players

    print("Error in key")
    return df_numeric

# Read data
DATA_TITLES = [
    '2021-2022 Football Team Stats',
    '2022-2023 Football Team Stats',
    '2021-2022 Football Player Stats',
    '2022-2023 Football Player Stats',
]

DFS = [get_numeric_df(pd.read_csv(DATA_PATH.joinpath(title + '.csv'), encoding='ISO-8859-1', delimiter=';'), title) for title in DATA_TITLES]

MEMOIZED_FIGS = {}
MEMOIZED_NUMERIC_DFS = {}
DATA_TYPE_LIST = ['Football Team Stats', 'Football Player Stats']
DATA_SEASON_LIST = ['2021-2022', '2022-2023']

COLUMN_NAMES = {
    "Rk" : "Rank",
    "Squad" : "Squad's name",
    "Country" : "Name of the country",
    "LgRk" : "Squad finish within the league",
    "MP" : "Matches played",
    "W" : "Wins",
    "D" : "Draws",
    "L" : "Losses",
    "GF" : "Goals for",
    "GA" : "Goals against",
    "GD" : "Goal difference",
    "Pts" : "Points",
    "Pts/MP" : "Points per game",
    "xG" : "Expected goals",
    "xGA" : "Expected goals allowed",
    "xGD" : "Expected goals difference",
    "xGD/90" : "Expected goals difference per 90 minutes",
    "Attendance" : "Attendance per game during this season, only for home matches",
    "Top Team Scorer" : "Top team scorer",
    "Goalkeeper" : "Goalkeeper",
}

df = pd.read_csv(DATA_PATH.joinpath("clinical_analytics.csv.gz"))

def generate_clustering_plot_teams(data_season):
    key = data_season + ' Football Team Stats'
    df_index = DATA_TITLES.index(key)
    df_numeric, squads = DFS[df_index]

    # Standardize and Normalize the data
    scaler = MinMaxScaler()  # Using MinMaxScaler for normalization
    df_scaled = scaler.fit_transform(df_numeric)

    # PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    principal_components = pca.fit_transform(df_scaled)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # Silhouette Score Analysis
    range_n_clusters = list(range(3, 11))
    silhouette_avg = []

    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(principal_df)
        silhouette_avg.append(silhouette_score(principal_df, cluster_labels))

    # Choose the optimal number of clusters
    optimal_clusters = range_n_clusters[silhouette_avg.index(max(silhouette_avg))]
    print("Optimal number of clusters:", optimal_clusters)

    # KMeans Clustering with Optimal Clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    principal_df['Cluster'] = kmeans.fit_predict(principal_df[['PC1', 'PC2']])

    # Add the team names back for visualization
    principal_df['Team'] = squads

    # Plot using Plotly
    fig = px.scatter(principal_df, x='PC1', y='PC2', color='Cluster', hover_data=['Team'])
    fig.update_layout(title='PCA and Clustering of Football Team Stats with Team Names on Hover')

    return fig

def generate_clustering_plot_teams_gaussian_mixture(data_season):
    key = data_season + ' Football Team Stats'
    df_index = DATA_TITLES.index(key)
    df_numeric, squads = DFS[df_index]

    # Standardize and Normalize the data
    scaler = MinMaxScaler()  # Using MinMaxScaler for normalization
    df_scaled = scaler.fit_transform(df_numeric)

    # PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    principal_components = pca.fit_transform(df_scaled)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # Silhouette Score Analysis
    range_n_clusters = list(range(3, 11))
    silhouette_avg = []

    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(principal_df)
        silhouette_avg.append(silhouette_score(principal_df, cluster_labels))

    # Choose the optimal number of clusters
    optimal_clusters = range_n_clusters[silhouette_avg.index(max(silhouette_avg))]
    print("Optimal number of clusters:", optimal_clusters)

    # KMeans Clustering with Optimal Clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    principal_df['Cluster'] = kmeans.fit_predict(principal_df[['PC1', 'PC2']])

    # Add the team names back for visualization
    principal_df['Team'] = squads

    # Plot using Plotly
    fig = px.scatter(principal_df, x='PC1', y='PC2', color='Cluster', hover_data=['Team'])
    fig.update_layout(title='PCA and Clustering of Football Team Stats with Team Names on Hover')

    return fig

def generate_clustering_plot_players(data_season):
    key = data_season + ' Football Player Stats'
    df_index = DATA_TITLES.index(key)
    df_numeric, players = DFS[df_index]

    # Standardize and Normalize the data
    scaler = MinMaxScaler()  # Using MinMaxScaler for normalization
    df_scaled = scaler.fit_transform(df_numeric)

    # PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    principal_components = pca.fit_transform(df_scaled)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # Silhouette Score Analysis
    range_n_clusters = list(range(3, 11))
    silhouette_avg = []

    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(principal_df)
        silhouette_avg.append(silhouette_score(principal_df, cluster_labels))

    # Choose the optimal number of clusters
    optimal_clusters = range_n_clusters[silhouette_avg.index(max(silhouette_avg))]
    print("Optimal number of clusters:", optimal_clusters)

    # KMeans Clustering with Optimal Clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    principal_df['Cluster'] = kmeans.fit_predict(principal_df[['PC1', 'PC2']])

    # Add the player names back for visualization
    principal_df['Player'] = players

    # Plot using Plotly
    fig = px.scatter(principal_df, x='PC1', y='PC2', color='Cluster', hover_data=['Player'])
    fig.update_layout(title='PCA and Clustering of Football Player Stats with Player Names on Hover')

    return fig

def generate_clustering_plot(data_type, data_season):
    if data_type == None or data_season == None:
        return px.scatter()
    key = data_season + ' ' + data_type
    if key+'/km' in MEMOIZED_FIGS.keys():
        return MEMOIZED_FIGS[key+'/km']
    
    fig = None
    if 'Team' in data_type:
        fig = generate_clustering_plot_teams(data_season)
    else:
        fig = generate_clustering_plot_players(data_season)

    # for optimization
    MEMOIZED_FIGS[key + '/km'] = fig
    return fig

def generate_clustering_plot_gaussian_mixture(data_type, data_season):
    if data_type == None or data_season == None:
        return px.scatter()
    key = data_season + ' ' + data_type
    if key+'/km' in MEMOIZED_FIGS.keys():
        return MEMOIZED_FIGS[key+'/km']
    
    fig = None
    if 'Team' in data_type:
        fig = generate_clustering_plot_teams_gaussian_mixture(data_season)
    else:
        fig = generate_clustering_plot_players(data_season)

    # for optimization
    MEMOIZED_FIGS[key + '/km'] = fig
    return fig

def generate_correlation_heatmap(data_type, data_season):
    if data_type == None or data_season == None:
        return px.imshow()

    key = data_season + ' ' + data_type
    if key+'/cor' in MEMOIZED_FIGS.keys():
        return MEMOIZED_FIGS[key+'/cor']

    df_numeric, _ = DFS[DATA_TITLES.index(key)]

    corr_matrix = df_numeric.corr()

    # Create a heatmap using Plotly Express
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
    fig.update_layout(title='Correlation Matrix of Football Team Statistics', 
                    xaxis_title='Variables', 
                    yaxis_title='Variables')
    
    MEMOIZED_FIGS[key + '/cor'] = fig
    return fig


def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("2021-2023 Football stats"),
            html.H3("Welcome to the Football Stats Analytics Dashboard"),
            html.Div(
                id="intro",
                children="Explore clinic patient volume by time of day, waiting time, and care score. Click on the heatmap to visualize patient experience at different time points.",
            ),
        ],
    )


def generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Select Data Type"),
            dcc.Dropdown(
                id="data-type-select",
                options=[{"label": i, "value": i} for i in DATA_TYPE_LIST],
                value=DATA_TYPE_LIST[0],
            ),
            html.Br(),
            html.P("Select Season"),
            dcc.Dropdown(
                id="data-season-select",
                options=[{"label": i, "value": i} for i in DATA_SEASON_LIST],
                value=DATA_SEASON_LIST[0],
            ),
            html.Br(),
            html.P("Choose Season Or Seasons"),
            html.Div([
                dcc.RangeSlider(
                    min=0,
                    max=20,
                    step=10,
                    marks={
                        0: '2021',
                        10: '2022',
                        20: '2023',
                    },
                    value=[0, 20]
                )
            ]),
        ],
    )

@app.callback(
    Output("football_correlation_heatmap", "figure"),
    [
        Input("data-type-select", "value"),
        Input("data-season-select", "value"),
    ],
)
def update_correlation_heatmap(data_type, data_season):
    return generate_correlation_heatmap(data_type, data_season)

@app.callback(
    Output("football_clusters_plot", "figure"),
    [
        Input("data-type-select", "value"),
        Input("data-season-select", "value"),
    ],
)
def update_clustering_plot(data_type, data_season):
    return generate_clustering_plot(data_type, data_season)

@app.callback(
    Output("football_gaussian_misxture", "figure"),
    [
        Input("data-type-select", "value"),
        Input("data-season-select", "value"),
    ],
)
def update_clustering_plot_gaussian_mixture(data_type, data_season):
    return px.scatter()
    # return generate_clustering_plot_gaussian_mixture(data_type, data_season)



app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Div("Mohamed LAHYANE, Yanis DENOUN \t"), html.Div(children=[html.A("https://github.com/yanisdenoun/dash-app")])],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                # Patient Volume Heatmap
                html.Div(
                    id="football_clusters_card",
                    children=[
                        html.B("Kmeans Clusters"),
                        html.Hr(),
                        dcc.Graph(id="football_clusters_plot"),
                    ],
                ),
                html.Div(
                    id="football_guassian_card",
                    children=[
                        html.B("Gaussian Misxture"),
                        html.Hr(),
                        dcc.Graph(id="football_gaussian_misxture"),
                    ],
                ),
                html.Div(
                    id="football_correlation_card",
                    children=[
                        html.B("Correlation Heatmap"),
                        html.Hr(),
                        dcc.Graph(id="football_correlation_heatmap"),
                        html.Hr(),
                    ],
                ),
            ],
        ),
    ],
)

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
