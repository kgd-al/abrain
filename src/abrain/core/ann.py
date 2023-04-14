from typing import Optional, Dict

import plotly.graph_objects as go

from .._cpp.phenotype import ANN, Point


def plotly_render(ann: ANN, labels: Optional[Dict[Point, str]] = None) \
        -> go.Figure:
    """
    Produce a 3D figure from an artificial neural network

    The returned figure can be used to save an interactive html session or a
    (probably poorly) render to e.g. a png file
    """

    n_type = ANN.Neuron.Type
    nodes = {t: [[], [], []] for t in [n_type.I, n_type.H, n_type.O]}
    edges = [[], [], []]

    names = {t: [] for t in nodes}

    for n in ann.neurons():
        x, y, z = n.pos.tuple()

        n_list = nodes[n.type]
        n_list[0] += [x]
        n_list[1] += [y]
        n_list[2] += [z]

        if labels is None or (label := labels.get(n.pos, None)) is None:
            label = f"{n.type.name}{len(names[n.type])}"
        names[n.type].append(label)

        for link in n.links():
            x2, y2, z2 = link.src().pos.tuple()
            edges[0] += [x, x2, None]
            edges[1] += [y, y2, None]
            edges[2] += [z, z2, None]

    trace_edges = go.Scatter3d(
        x=edges[0], y=edges[1], z=edges[2],
        mode='lines', line=dict(color='black', width=1), opacity=.1,
        hoverinfo='none', showlegend=False
    )

    trace_nodes = []
    for n_type, name, color in [(n_type.I, 'Inputs',  'black'),
                                (n_type.O, 'Outputs', 'black'),
                                (n_type.H, 'Hidden',  'gray')]:
        n_list = nodes[n_type]
        trace_nodes.append(
            go.Scatter3d(
                x=n_list[0], y=n_list[1], z=n_list[2],
                mode='markers',
                marker=dict(symbol='circle',
                            size=10,
                            color=color),
                name=name,
                hovertemplate='<b>' + name + '</b><br>'
                              '<i>%{hovertext}</i><br>'
                              'x: %{x}<br>'
                              'y: %{y}<br>'
                              'z: %{z}<extra></extra>',
                hovertext=names[n_type]
            )
        )

    data = [trace_edges, *trace_nodes]
    fig = go.Figure(data=data)

    camera = dict(
        eye=dict(x=-2, y=0, z=1)
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=3, range=[-1.1, 1.1]),
            yaxis=dict(nticks=3, range=[-1.1, 1.1]),
            zaxis=dict(nticks=3, range=[-1.1, 1.1])
        ), scene_aspectmode='cube',
        scene_camera=camera,
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Courrier"
        )
    )

    return fig
