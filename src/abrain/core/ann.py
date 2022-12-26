import plotly.graph_objects as go

from .._cpp.phenotype import ANN
from .._cpp.phenotype import Point  # noqa For transparent use from importers


def plotly_render(ann: ANN) -> go.Figure:
    """
    Produce a 3D figure from an artificial neural network

    The returned figure can be used to save an interactive html session or a
    (probably poorly) render to e.g. a png file
    """
    io_nodes = [[], [], []]
    hd_nodes = [[], [], []]
    edges = [[], [], []]
    for n in ann.neurons():
        x, y, z = n.pos.tuple()

        nodes = hd_nodes if n.type == ANN.Neuron.Type.H else io_nodes
        nodes[0] += [x]
        nodes[1] += [y]
        nodes[2] += [z]

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
    for nodes, name, color in [(io_nodes, 'inputs/outputs', 'black'),
                               (hd_nodes, 'hidden neurons', 'gray')]:
        trace_nodes.append(
            go.Scatter3d(
                x=nodes[0], y=nodes[1], z=nodes[2],
                mode='markers',
                marker=dict(symbol='circle',
                            size=10,
                            color=color),
                name=name
            )
        )

    data = [trace_edges, *trace_nodes]
    fig = go.Figure(data=data)

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=3, range=[-1.1, 1.1]),
            yaxis=dict(nticks=3, range=[-1.1, 1.1]),
            zaxis=dict(nticks=3, range=[-1.1, 1.1])
        ), scene_aspectmode='cube',
        showlegend=False
    )

    return fig
