#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive 2D molecular structure visualisation application
Using T-SNE for dimensionality reduction
@author: Yu Che
"""
import dash
import flask
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

df = pd.read_csv('../SMP_project/tsne_data.csv')
axis_template = dict(
    showgrid=False,
    zeroline=False,
    showline=False,
    showticklabels=False,
)
app.layout = html.Div([
    html.Div(
        dcc.Graph(
            id='clickable_plot',
            figure={
                'data': [dict(
                    x=df.loc[:, 'pos0'],
                    y=df.loc[:, 'pos1'],
                    text=df.loc[:, 'Molecules'],
                    mode='markers',
                    marker=dict(
                        size=5,
                        colorbar=dict(
                            thicknessmode='pixels',
                            thickness=20,
                            title='HER Ave umol/h'
                        ),
                        color=df.loc[:, 'HER Ave umol/h'],
                        colorscale='RdBu',
                        showscale=True
                    )
                )],
                'layout': go.Layout(
                    clickmode='event+select',
                    title='SMP_TSNE_map',
                    hovermode='closest',
                    xaxis=axis_template,
                    yaxis=axis_template,
                    showlegend=False,
                    width=960,
                    height=540
                )
            })
    ),
    html.Div([
        html.Div(id='hover_molecule'),
        html.Div(id='selected_molecule')]
    )
])


def images_component(data):
    image_name = 'template_image'
    html_images = []

    def html_image(name):
        one_image = html.Img(
            src='https://res.cloudinary.com/yucheimages/image/upload'
                '/v1560534817/molecular_images/{}.png'.format(name),
            style={'width': '400px', 'height': '300px'}
        )
        return one_image

    try:
        number_molecules = len(data['points'])
        for i in range(number_molecules):
            index = int(data['points'][i]['pointIndex'])
            image_name = df.loc[index, 'Molecules']
            html_images.append(html_image(image_name))
    except TypeError:
        html_images.append(html_image(image_name))
    return html.Div(html_images)


@app.callback(dash.dependencies.Output('hover_molecule', 'children'),
              [dash.dependencies.Input('clickable_plot', 'hoverData')])
def display_hover_image(hoverData):
    return images_component(hoverData)


@app.callback(dash.dependencies.Output('selected_molecule', 'children'),
              [dash.dependencies.Input('clickable_plot', 'selectedData')])
def display_selected_data(selectedData):
    return images_component(selectedData)


if __name__ == '__main__':
    app.run_server(host='0.0.0.0')
