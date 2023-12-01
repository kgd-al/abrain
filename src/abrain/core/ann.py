import logging
import pprint
from enum import Flag, auto
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import plotly.graph_objects as go

from .._cpp.phenotype import ANN, Point


logger = logging.getLogger(__name__)


def plotly_render(ann: ANN, labels: Optional[Dict[Point, str]] = None) \
        -> go.Figure:
    """
    Produce a 3D figure from an artificial neural network

    The returned figure can be used to save an interactive html session or a
    (probably poorly) rendering to e.g. a png file
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


class ANNMonitor:
    def __init__(self, ann: ANN, labels: dict[Point, str],
                 folder: Path,
                 neurons_file: Optional[Path],
                 dynamics_file: Optional[Path]):
        self.ann = ann
        self.labels = labels

        self.save_folder = folder
        if neurons_file:
            self.neural_data_file = folder.joinpath(neurons_file)

        self.neurons_data = None
        if neurons_file or dynamics_file:
            self.neurons_data = pd.DataFrame(
                columns=[f"{n.pos}:{self.labels.get(n.pos)}"
                         for n in self.ann.neurons()]
            )

        self.axonal_data = None
        if dynamics_file:
            self.interactive_plot_file = folder.joinpath(dynamics_file)
            self.axonal_data = pd.DataFrame(
                columns=[f"{l_.src().pos}->{n.pos}"
                         for n in self.ann.neurons() for l_ in n.links()]
            )
            # pprint.pprint(self.axonal_data.columns)

    def open(self):
        pass

    def step(self):
        if self.neurons_data is not None:
            self.neurons_data.loc[len(self.neurons_data)] = [
                n.value for n in self.ann.neurons()
            ]
        if self.axonal_data is not None:
            self.axonal_data.loc[len(self.axonal_data)] = [
                l_.src().value * l_.weight
                for n in self.ann.neurons() for l_ in n.links()
            ]

    def close(self):
        if self.neural_data_file:
            self.neurons_data.to_csv(self.neural_data_file, sep=' ')
            logger.info(f"Generated {self.neural_data_file}")

        if self.interactive_plot_file:
            # print(self.axonal_data)

            x, y, z = zip(*[n.pos.tuple() for n in self.ann.neurons()])
            # fig = go.Figure(
            #     data=[go.Scatter3d(x=x, y=y, z=z,
            #                        mode="markers",
            #                        marker=dict(color="red", size=10))]
            # )

            fig = plotly_render(self.ann, self.labels)

            # print([row for row in self.neurons_data.iterrows()])
            frames = [
                go.Frame(
                    data=[
                        go.Scatter3d(
                            x=x, y=y, z=z,
                            mode='markers',
                            marker=dict(
                                size=12,
                                color=row,
                                colorscale='RdBu_r',
                                colorbar=dict(
                                    title="Colorbar"
                                )
                            )
                        )
                    ],
                    name=f'frame{index}'
                ) for index, row in self.neurons_data.iterrows()
            ]
            fig.update(frames=frames)

            def frame_args(duration):
                return {
                    "frame": {"duration": duration},
                    "mode": "immediate",
                    "fromcurrent": True,
                    "transition": {"duration": duration, "easing": "linear"},
                }

            sliders = [
                {"pad": {"b": 10, "t": 60},
                 "len": 0.9,
                 "x": 0.1,
                 "y": 0,

                 "steps": [
                     {"args": [[f.name], frame_args(0)],
                      "label": str(k),
                      "method": "animate",
                      } for k, f in enumerate(fig.frames)
                 ]
                 }
            ]

            fig.update_layout(

                updatemenus=[{"buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "Pause",
                        "method": "animate",
                    }],

                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
                ],
                sliders=sliders
            )

            fig.write_html(self.interactive_plot_file,
                           auto_play=False)
            logger.info(f"Generated {self.interactive_plot_file}")
            # fig.show()
            exit(42)

        # print("[kgd-debug] testing pyplot dynamical rendering")
