#!/usr/bin/env python3
"""
-*- coding: utf-8 -*-
Author:Yu Che
Dash 2D&3D plot version 1.4
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import pymongo
import multiprocessing

# Dash format and connecting to mongoDB
app = dash.Dash('ESF maps')
client = pymongo.MongoClient('mongodb://138.253.124.96/')
client.users.authenticate('user1', '1234')
db = client.users


def read(dic):
    return dic


# Retrieval data from mongoDB
p = multiprocessing.Pool()
result = p.map(
    read, [item for item in db.ESF.find({'job_number': {'$regex': 'T2*'}})]
)
p.close()
df = pd.DataFrame(data=result)
data_columns = df.columns

# All HTML elements
app.layout = html.Div([
    # Dash title and icon
    html.Div([
        html.Img(src="https://s3-us-west-1.amazonaws.com/plotly-tutorials"
                     "/logo/new-branding/dash-logo-by-plotly-stripe.png",
                 style={'float': 'right', 'position': 'relative',
                        'height': '60px', 'bottom': '10px', 'left': '20px'}),
        html.H2('Dash for ESF maps',
                style={'position': 'relative', 'display': 'inline',
                       'top': '0px', 'left': '10px',
                       'font-family': 'Dosis', 'font-size': '3.0rem',
                       'color': '#3D4B56'})
    ], className='plot_title', style={'position': 'relative', 'right': '15px'}
    ),
    # Graph type selection, using dash radio items components
    dcc.Graph(id='indicatorgraphic'),
    dcc.RadioItems(
        id='plot_type',
        options=[
            {'label': '2D Scatters', 'value': '2D'},
            {'label': '3D Scatters', 'value': '3D'},
        ], value='3D', style={'display': 'inline-block'}
     ),
    # XYZ axises selection, using dash dropdown elements
    html.Div([
        html.Div([
            html.P('X-axis:'),
            dcc.Dropdown(
                id='x_axis_column',
                options=[{'label': i, 'value': i} for i in data_columns],
                value='Density'
            )
        ], style={'width': '20%', 'display': 'inline-block'}
        ),
        html.Div([
            html.P('Y-axis:'),
            dcc.Dropdown(
                id='y_axis_column',
                options=[{'label': i, 'value': i} for i in data_columns],
                value='Lattice_energy'
            )
        ], style={'width': '20%', 'display': 'inline-block'}
        ),
        html.Div([
            html.P('Z-axis:'),
            dcc.Dropdown(
                id='z_axis_column',
                options=[{'label': i, 'value': i} for i in data_columns],
                value='Structure'
            )
        ], style={'width': '20%', 'display': 'inline-block'}
        )
    ], className='axes'
    ),
    # Color bar properties selection, using dash dropdown elements
    html.Div([
        html.P('Colour_bar:'),
        dcc.Dropdown(
            id='colour_column',
            options=[{'label': i, 'value': i} for i in data_columns],
            value='Density'
        )
    ], style={'width': '20%', 'display': 'inline-block'}
    ),
    # Data range selection, using dash range slider elements
    html.Div([
        html.P('Range_slider:'),
        dcc.Dropdown(
            id='range_column',
            options=[{'label': i, 'value': i} for i in data_columns],
            value='Unitcell_volume'
        )
    ], style={'width': '20%', 'display': 'inline-block'}
    ),
    # Print text to present data range
    html.Div([
        html.P('Select data range:'),
        dcc.RangeSlider(id='range_slider')
    ], style={'width': '60%'}
    ),
    html.Div(id='selected_data')
])


# Setting range slider properties(min, max and steps)
@app.callback(
    dash.dependencies.Output('range_slider', 'min'),
    [dash.dependencies.Input('range_column', 'value')])
def select_bar1(range_column_value):
    return df[range_column_value].min()


@app.callback(
    dash.dependencies.Output('range_slider', 'max'),
    [dash.dependencies.Input('range_column', 'value')])
def select_bar2(range_column_value):
    return df[range_column_value].max()


@app.callback(
    dash.dependencies.Output('range_slider', 'value'),
    [dash.dependencies.Input('range_column', 'value')])
def select_bar3(range_column_value):
    return [df[range_column_value].min(), df[range_column_value].max()]


# Plot graph controlled by dropdown and range slider
@app.callback(
    dash.dependencies.Output('indicatorgraphic', 'figure'),
    [dash.dependencies.Input('plot_type', 'value'),
     dash.dependencies.Input('x_axis_column', 'value'),
     dash.dependencies.Input('y_axis_column', 'value'),
     dash.dependencies.Input('z_axis_column', 'value'),
     dash.dependencies.Input('colour_column', 'value'),
     dash.dependencies.Input('range_column', 'value'),
     dash.dependencies.Input('range_slider', 'value')])
def update_graph(plot_type_value, x_axis_column_name, y_axis_column_name,
                 z_axis_column_name, colour_column_value, range_column_value,
                 range_slider_value):
    filtered_df = pd.DataFrame(
        data=df[(df[range_column_value] > range_slider_value[0]) &
                (df[range_column_value] < range_slider_value[1])]
    )
    # 2D scatter plot
    if plot_type_value == '2D':
        return {
            'data': [go.Scattergl(
                x=filtered_df[x_axis_column_name],
                y=filtered_df[y_axis_column_name],
                text=filtered_df['Structure_name'],
                mode='markers',
                marker={'size': 10,
                        'color': filtered_df[colour_column_value],
                        'opacity': 0.8,
                        'line': {'color': 'rgb(240, 240, 240)', 'width': 0.5},
                        'colorbar': {'title': colour_column_value},
                        'colorscale': 'Viridis',
                        'showscale': True}
            )],
            'layout': go.Layout(
                height=800,
                xaxis={'title': x_axis_column_name, 'zeroline': True},
                yaxis={'title': y_axis_column_name, 'zeroline': True},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                hovermode='closest'
            )
        }
    # 3D scatter plot
    elif plot_type_value == '3D':
        return {
            'data': [go.Scatter3d(
                x=filtered_df[x_axis_column_name],
                y=filtered_df[y_axis_column_name],
                z=filtered_df[z_axis_column_name],
                text=filtered_df['Structure_name'],
                mode='markers',
                marker={'size': 5,
                        'color': filtered_df[colour_column_value],
                        'colorbar': {'title': colour_column_value},
                        'colorscale': 'RdBu',
                        'showscale': True}
            )],
            'layout': go.Layout(
                height=800,
                scene=dict(
                    xaxis={'title': x_axis_column_name, 'zeroline': True},
                    yaxis={'title': y_axis_column_name, 'zeroline': True},
                    zaxis={'title': z_axis_column_name, 'zeroline': True},
                ),
                margin={'l': 40, 'b': 40, 't': 10, 'r': 0}
            )
        }


# Print range slider ranges
@app.callback(
    dash.dependencies.Output('selected_data', 'children'),
    [dash.dependencies.Input('range_slider', 'value'),
     dash.dependencies.Input('range_column', 'value')])
def callback(range_slider_value, range_column_value):
    return 'You have selected "{0}" range of {1}'.format(
        range_slider_value, range_column_value)


if __name__ == '__main__':
    app.run_server()
