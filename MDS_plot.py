#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime
from sklearn import manifold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AffinityPropagation


# noinspection PyRedundantParentheses
class MdsPlot():
    """
    This class is used to proceed a plot followed with clustering and
    scaling methods.
    """
    def __init__(self):
        self.df = pd.DataFrame()
        self.data_df = pd.DataFrame()
        self.selected_df = pd.DataFrame()
        self.descriptor = None
        self.similarities = None
        self.features = None
        self.size = None
        self.pos_df = pd.DataFrame()

    def data_retrieve(self, path, size, descriptor):
        """
        Data frame retrieve function.

        :param path: The path for input data file.
        :param size: The range of columns in data frame.
        :param descriptor: The variable will be used in plot.
        :type path: str
        :type size: tuple
        :type descriptor: str
        :return: None
        """
        print('Reading data frame...')
        self.size = size
        # Reading data frame
        file_format = path.split('.')[-1]
        if file_format == 'csv':
            self.df = pd.read_csv(path)
        elif file_format == ('xls' or 'xlsx'):
            self.df = pd.read_excel(path)
        elif file_format == 'pkl':
            self.df = pd.read_pickle(path)
        else:
            print('Error! Please input the correct format file.')
            exit()
        # Selecting the range of input data
        self.data_df = self.df.iloc[:, size[0]:size[1]]
        self.descriptor = pd.DataFrame(
            self.df[['Structure', 'Final_lattice_E', descriptor]])
        # Delete no numerical data to avoid error in normalization
        for column in self.data_df.columns:
            if self.data_df.dtypes[column] != ('float64' or 'int64'):
                self.data_df = self.data_df.drop(column, axis=1)
        print('Normalizing...')
        # Normalization
        data = self.data_df.values
        self.features = MinMaxScaler().fit_transform(data)
        print(
            'Finished!\n'
            'Selected data frame: self.data_df\n'
            'Descriptors matrix:  self.features\n'
            'Descriptor shape:    {}'.format(self.features.shape)
        )

    def affinity_propagation_cluster(self):
        """
        Using affinity propagation to cluster the data frame. The cluster
        information is written in the 'AffinityPropagation' column.

        :return: None
        """
        print('Affinity propagation starting...')
        start = datetime.now()
        af = AffinityPropagation(
            max_iter=5000, convergence_iter=50, preference=None
        ).fit(self.features)
        cluster_centers_indices = af.cluster_centers_indices_
        df_labels = pd.DataFrame(af.labels_)
        n_clusters = len(cluster_centers_indices)
        # Merge cluster information into origin data frame
        self.data_df = self.data_df.merge(
            df_labels, left_index=True, right_index=True
        )
        self.data_df.rename(columns={0: 'AffinityPropagation'}, inplace=True)
        self.data_df = self.data_df.merge(
            self.descriptor, left_index=True, right_index=True)
        print(
            'Finished!\n'
            'Estimated number of clusters: {}\n'
            'Total time:{}'.format(n_clusters, (datetime.now() - start))
        )

    def cluster_structure_selection(self, descriptor):
        """
        Choosing the lowest value for selected descriptor in each cluster.

        :param descriptor: One selected variable in the data frame.
        :type descriptor: str
        :return: None
        """
        # The lowest lattice energy for each cluster
        for i in range(0, self.data_df.AffinityPropagation.max() + 1):
            df_cluster = self.data_df[self.data_df['AffinityPropagation'] == i]
            selected_structure = df_cluster[
                df_cluster[descriptor] == df_cluster[descriptor].min()
                ]
            self.selected_df = self.selected_df.append(selected_structure)
        # Rearrange index
        self.selected_df.index = range(len(self.selected_df))

    def mds_calculation(self):
        """
        Using multidimensional scaling for the selected data to calculate
        2D coordinators. The coordinator information is written in 'pos0' and
        'pos1' columns.

        :return: None
        """
        print('Distance calculation...')
        start = datetime.now()
        features = self.selected_df.iloc[
                   :, self.size[0]:(self.size[1] - 1)].values
        scaled_features = MinMaxScaler().fit_transform(features)
        distance = euclidean_distances(scaled_features)
        self.similarities = distance / distance.max()
        print('Multidimensional scaling starting...')
        seed = np.random.RandomState(seed=0)
        mds = manifold.MDS(
            n_components=2, max_iter=30000, random_state=seed,
            eps=1e-12, dissimilarity="precomputed", n_jobs=1
        )
        pos = mds.fit(self.similarities).embedding_
        self.pos_df = pd.DataFrame(data=pos, columns=['pos0', 'pos1'])
        self.selected_df = self.selected_df.merge(
            self.pos_df, left_index=True, right_index=True
        )
        print(
            'Finished.\n'
            'Distance matrix:     self.similarities\n'
            'Selected data frame: self.selected_df\n'
            'Total time:{}'.format(datetime.now() - start)
        )

    def plot(self, title, size, colour, tag=(), lines=False, text='Structure'):
        """
        This function is based on plotly API to generate a scatter plot for
        multidimensional scaling.
        Using plotly function to generate the plot in notebook.

        :param title: The plot title.
        :param text:  The descriptor will be shown in the text part in plot.
        :param size: The name of a descriptor to scale the scatter size.
        :param colour: The name of a descriptor to scale the colour.
        :param lines: Enable lines to describe the similarity.
        :param tag: The name of selected points shown in diamond style.
        :type title: str
        :type size: str
        :type colour: str
        :type tag: tuple
        :return: Plotly figure object.
        """
        tag_list = list(
            self.selected_df[self.selected_df.Structure.isin(tag)].index
        )
        # Plot the network line part:
        if lines:
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=0.5, color='rgb(134, 134, 134)'),
                opacity=0.7,
                hoverinfo='none',
                mode='lines'
            )
            # Generate the start-stop point coordinates
            segments = []
            n_space = len(self.similarities)
            for i in range(n_space):
                for j in range(n_space):
                    # Limited the line length
                    if 0.03 < self.similarities[i, j] < 0.15:
                        p1 = [self.pos_df.iloc[i, 0], self.pos_df.iloc[i, 1]]
                        p2 = [self.pos_df.iloc[j, 0], self.pos_df.iloc[j, 1]]
                        segments.append([p1, p2])
            # Adding each line's coordinates in plot
            for edge in segments:
                x0, y0 = edge[0]
                x1, y1 = edge[1]
                edge_trace['x'] += tuple([x0, x1, None])
                edge_trace['y'] += tuple([y0, y1, None])
        # Scatter plot
        trace0 = go.Scatter(
            x=self.selected_df.loc[:, 'pos0'],
            y=self.selected_df.loc[:, 'pos1'],
            text=self.selected_df.loc[:, text],
            mode='markers',
            selectedpoints=tag_list,
            selected=dict(marker=dict(opacity=0)),
            unselected=dict(marker=dict(opacity=1)),
            marker=dict(
                symbol='circle',
                size=(8 + self.selected_df.loc[:, size]),
                color=self.selected_df.loc[:, colour],
                colorscale='RdBu',
                colorbar=dict(
                    thicknessmode='pixels',
                    thickness=20,
                    title='Lattice energy'
                ),
                reversescale=True,
                showscale=True
            )
        )
        trace1 = go.Scatter(
            x=self.selected_df.loc[:, 'pos0'],
            y=self.selected_df.loc[:, 'pos1'],
            text=self.selected_df.loc[:, text],
            mode='markers',
            selectedpoints=tag_list,
            selected=dict(marker=dict(opacity=1)),
            unselected=dict(marker=dict(opacity=0)),
            marker=dict(
                symbol='diamond',
                size=(8 + self.selected_df.loc[:, size]),
                color=self.selected_df.loc[:, colour],
                colorscale='RdBu',
                reversescale=True,
                showscale=False
            )
        )
        # Hidden all axis
        axis_template = dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
        )
        layout = go.Layout(
            title=title,
            hovermode='closest',
            xaxis=axis_template,
            yaxis=axis_template,
            showlegend=False
        )
        # The plot function
        plot_fig = go.Figure(data=[trace0, trace1], layout=layout)
        return plot_fig


if __name__ == '__main__':
    # Test script
    plot = MdsPlot()
    plot.data_retrieve(
        path='../4EPK/T2.csv', size=(None, 29), descriptor='CH4_Del(65-5.8bar)'
    )
    plot.affinity_propagation_cluster()
    plot.cluster_structure_selection(descriptor='Final_lattice_E')
    plot.mds_calculation()
    fig = plot.plot(
        title='T2 MDS plot', size='CH4_Del(65-5.8bar)',
        colour='Final_lattice_E', text='Structure',
        tag=('job_00014', 'job_00054', 'job_00120', 'job_00186')
    )
