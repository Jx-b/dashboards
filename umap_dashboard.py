#!/usr/bin/env python3
# -*- codinchmodg: utf-8 -*-

# Imports
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import umap


class Dash_UMAP(dash.Dash):
    
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    
    def __init__(self, data, labels):
        super().__init__(__name__, external_stylesheets= self.external_stylesheets)
        self.title = 'UMAP analysis'
        self.data = data
        self.labels = labels
        self.labelled_data = pd.DataFrame.join(data, labels)

        #Create layout
        self.layout = html.Div([    
            
            html.Div([
                html.Div([
                    html.H4('Feature scaling (zscore)'),
                    dcc.RadioItems(
                        id= 'scaling_radio',
                        options=[
                            {'label': 'On', 'value': 'On'},
                            {'label': 'Off', 'value': 'Off'}
                        ],
                        value= 'On',
                        labelStyle={'display': 'inline-block'}
                    ),], style= {'grid-area': 'control1'}
                 ),
                
                html.Div([
                    html.H4('UMAP parameters:'),
                    html.H6(id='neighbors_label', children= ['Number of neighbours: 15']),
                    dcc.Slider(
                        id = 'n_neigbors_slider',
                        min=-2,
                        max=int(len(self.labelled_data)/2),
                        step=1,
                        value=15,
                        updatemode= 'drag'
                    ),
                    
                    html.H6(id='dist_label', children= ['Minimum distance: 0.1']),
                    dcc.Slider(
                        id = 'min_dist_slider',
                        min=-0,
                        max=1,
                        step=0.05,
                        value=0.1,
                        updatemode= 'drag'
                    ),
                    
                    html.H6('Metric:'),
                    dcc.Dropdown(
                        id = 'metric_dropdown',
                        options= [
                                {'label': 'euclidean', 'value': 'euclidean'},
                                {'label': 'manhattan', 'value': 'manhattan'},
                                {'label': 'chebyshev', 'value': 'chebyshev'},
                                {'label': 'minkowski', 'value': 'minkowski'},
                                {'label': 'canberra', 'value': 'canberra'},
                                {'label': 'braycurtis', 'value': 'braycurtis'},
                                {'label': 'haversine', 'value': 'haversine'},
                                {'label': 'mahalanobis', 'value': 'mahalanobis'},
                                {'label': 'wminkowski', 'value': 'wminkowski'},
                                {'label': 'seuclidean', 'value': 'seuclidean'},
                                {'label': 'cosine', 'value': 'cosine'},
                                {'label': 'correlation', 'value': 'correlation'}
                        ],
                        value='euclidean',
                    )
                    ], style= {'grid-area': 'control2'}
                ),
                html.Div([
                    html.Button('Recompute UMAP', 
                                id='umap_button',
                                title= 'Click to compute UMAP with new parameters',
                                disabled= False
                               )
                    ], style= {'grid-area': 'control3', 'place-self':'center'}
                ),
                
                html.Div([
                    html.H6('Color by'),
                    dcc.Dropdown(
                        id = 'colorby_dropdown',
                        options= [{'label': group, 'value': group} for group in self.labelled_data.columns],
                        value= self.labels.columns[0],   
                        )
                    ], style= {'grid-area': 'control4'}
                ),
                
                dcc.Store(id='umap_storage', data={})
                ],
                style= {'grid-area': 'controls',
                        'display': 'grid', 'grid-template-rows': '1fr 2fr 1fr 1fr', 'grid-template-areas': '"control1" "control2" "control3" "control4"'}
            ),
            dcc.Loading(                    
                children=[
                    dcc.Graph(
                        id='scatter',
                        figure= self.plot_umap(self.perform_umap(self.normalise(self.data)),self.labels.iloc[:,0]),
                        config=dict(showSendToCloud=True),
                        style= {'height': '100%'}
                    ),
                ],
                type='dot',
                style= {'grid-area': 'scatter-plot'}
            ),

            
            html.Div([
                html.H4(children='Data table'),
                html.Div(children= self.generate_table(self.labels), id='table')
                ], 
                style= {'grid-area': 'table'}
            )
            
            ], style= {'display': 'grid', 
                    'grid-template-columns': '1fr 2fr', 
                    'grid-template-areas': '"controls scatter-plot" "table table"',
                    'margin-left': '5%',
                    'margin-right': '5%'
                    }
    )
        
    def normalise(self, data):
        return pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns)
    
    def perform_umap(self, data, n_neighbors= 15, min_dist= 0.1, metric= 'euclidean'):
        reducer = umap.UMAP(n_neighbors= n_neighbors, min_dist= min_dist, metric= metric, n_components= 2)
        embedding = pd.DataFrame(reducer.fit_transform(data), index= data.index)
        return embedding
     
    def plot_group_umap(self, embedding, color_by):
        """Plots a scatter plot of UMAP embedding with dots colored by group"""
        umap_data = pd.DataFrame.join(embedding, color_by)
        data= [go.Scatter(
                        x= group.iloc[:,0],
                        y= group.iloc[:,1],
                        text= group.index,
                        mode='markers',
                        marker= {
                            'size': 10,
                            'color': idx,
                            'opacity': .8,
                        },
                        name= name) for idx,(name, group) in enumerate(umap_data.groupby(color_by))]
        return data
    
    def plot_heatmap_umap(self, embedding, color_by):
        """Plots a scatter plot of UMAP embedding with dots colored by value"""
        data=  [go.Scatter(
                        x= embedding.iloc[:,0],
                        y= embedding.iloc[:,1],
                        text= embedding.index,
                        mode='markers',
                        marker= {
                            'size': 10,
                            'color': color_by,
                            'colorscale': 'Viridis',
                            'opacity': .8,
                        },
                        name= color_by.name
                    )
                ]
        return data
    
    def plot_umap(self, embedding, color_by):
        """Plots a scatter plot of UMAP embedding"""
        if np.issubdtype(color_by.dtype, np.number):
            data = self.plot_heatmap_umap(embedding, color_by)
        else:
            data = self.plot_group_umap(embedding, color_by)
        layout = go.Layout(
                            title= 'UMAP',
                            xaxis = dict(zeroline = False),
                            yaxis = dict(zeroline = False),
                            clickmode= 'event+select',
                            dragmode= 'select',
                            uirevision= True
                        )
        return {'data': data, 'layout':layout}
    
    def generate_table(self, df):
        """Generate a html table displaying the content of a pandas dataframe"""
        dataframe = df.reset_index()
        table = dash_table.DataTable(
            columns= [{"name": i, "id": i, 'deletable': True} for i in dataframe.columns],
            data= dataframe.to_dict("rows"),
            sorting=True,
            filtering=True,
            sorting_type='multi',
            style_as_list_view=True,
            style_cell={'padding': '5px'},
            style_header={
                'fontWeight': 'bold'
                },
            n_fixed_rows= 2,
            )

        return table

def run_dashboard(data, labels):
    """Creates a dashboard for visual exploration of UMAP analysis
    
    data: a pandas dataFrame whose columns contain the features on which to perform PCA
    labels: a pandas dataFrame whose columns contain the labels, the index must be identical to data
    """
    ### Create App ###
    app = Dash_UMAP(data, labels)
    
    ### Add Interactivity ###
    @app.callback(
        dash.dependencies.Output('neighbors_label', 'children'),
        [dash.dependencies.Input('n_neigbors_slider', 'value')])
    def update_slider1(slider_value):
        return ['Number of neighbors: ' + str(slider_value)]
    
    @app.callback(
        dash.dependencies.Output('dist_label', 'children'),
        [dash.dependencies.Input('min_dist_slider', 'value')])
    def update_slider2(slider_value):
        return ['Minimum distance: ' + str(slider_value)]
    
    @app.callback(
        dash.dependencies.Output('umap_storage', 'data'),
        [dash.dependencies.Input('umap_button', 'n_clicks')],
        [dash.dependencies.State('scaling_radio', 'value'),
         dash.dependencies.State('n_neigbors_slider', 'value'),
         dash.dependencies.State('min_dist_slider', 'value'),
         dash.dependencies.State('metric_dropdown', 'value')])
    def update_umap(n_clicks, scaling, n_neighbors, min_dist, metric):
        norm_data = app.normalise(app.data) if scaling == 'On' else app.data
        embedding = app.perform_umap(norm_data, n_neighbors, min_dist, metric)
        return embedding.to_json()
    
    @app.callback(
        dash.dependencies.Output('scatter', 'figure'),
        [dash.dependencies.Input('umap_storage', 'data'), dash.dependencies.Input('colorby_dropdown', 'value')])
    def update_scatter(data, color_by):
        embedding = pd.read_json(data)
        return app.plot_umap(embedding, app.labelled_data.loc[:,color_by])
    
    @app.callback(
        dash.dependencies.Output('table', 'children'),
        [dash.dependencies.Input('scatter', 'selectedData')])
    def update_table(selectedData):
        if selectedData:
            selected = [point['text'] for point in selectedData['points']]
            
            return app.generate_table(app.labels[app.labels.index.isin(selected)])
        else:
            return app.generate_table(app.labels)
    
    app.run_server(debug=True)

if __name__ == '__main__':
    
    # Import data
    iris_data = datasets.load_iris()
    data = pd.DataFrame(iris_data['data'], columns= iris_data['feature_names'])
    labels = pd.DataFrame(iris_data['target'], columns = ['class']).apply(lambda x : iris_data['target_names'][x])
    labels['class_num'] = iris_data['target']

    # Create App
    run_dashboard(data, labels)
