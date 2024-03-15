import logging
import math
from itertools import chain
from pathlib import Path
from typing import Optional, Dict, List, Iterable, Union

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

    return _figure(data=[_edges(ann), _neurons(ann, labels)])


def _iter_neurons(ann: ANN):
    for n in ann.neurons():
        yield n


def _iter_axons(ann: ANN):
    for dst in ann.neurons():
        for link in dst.links():
            yield link, link.src(), dst


class ANNMonitor:
    def __init__(self, ann: ANN, labels: Optional[Dict[Point, str]],
                 folder: Path,
                 neurons_file: Optional[Union[Path, str]],
                 dynamics_file: Optional[Union[Path, str]],
                 dt: Optional[float] = None):
        self.ann = ann

        self.labels = labels

        self.dt = dt

        self.save_folder = folder
        self.neural_data_file = None
        if neurons_file:
            self.neural_data_file = folder.joinpath(neurons_file)

        if self.labels is None:
            def _id(_n): return str(_n.pos)
        else:
            def _id(n): return f"{n.pos}:{self.labels.get(n.pos)}"

        self.neurons_data = None
        if neurons_file or dynamics_file:
            self.neurons_data = _TinyDataFrame(
                columns=[_id(n) for n in _iter_neurons(self.ann)]
            )

        self.axonal_data = None
        self.axonal_data_file = None
        if dynamics_file:
            self.axonal_data_file = folder.joinpath(dynamics_file)
            self.axonal_data = _TinyDataFrame(
                columns=[f"{src.pos}->{dst.pos}"
                         for _, src, dst in _iter_axons(self.ann)]
            )
            # pprint.pprint(self.axonal_data.columns)

    def step(self):
        if self.neurons_data is not None:
            self.neurons_data.append([
                n.value for n in _iter_neurons(self.ann)
            ])
        if self.axonal_data is not None:
            self.axonal_data.append([
                src.value * link.weight
                for link, src, dst in _iter_axons(self.ann)
            ])

    def close(self):
        if self.neural_data_file:
            self.neurons_data.to_csv(self.neural_data_file, sep=' ')
            logger.info(f"Generated {self.neural_data_file}")

        if self.axonal_data_file:
            def c_range(array):
                v_min, v_max = math.inf, -math.inf
                for v in array:
                    if isinstance(v, Iterable):
                        _v_min, _v_max = c_range(v)
                        if _v_min < v_min:
                            v_min = _v_min
                        if v_max < _v_max:
                            v_max = _v_max
                    else:
                        if v < v_min:
                            v_min = v
                        if v_max < v:
                            v_max = v
                v_min = min([v_min, -v_min, v_max, -v_max])
                return [v_min, -v_min]

            nv_min, nv_max = c_range(self.neurons_data.data)
            # print(f"[kgd-debug] {nv_min=}, {nv_max=}")

            ew_min, ew_max = c_range(
                [link.weight for link, _, _ in _iter_axons(self.ann)])
            # print(f"[kgd-debug] {ew_min=}, {ew_max=}")

            ev_min, ev_max = c_range(self.axonal_data.data)
            # print(f"[kgd-debug] {ev_min=}, {ev_max=}")

            frames = [
                go.Frame(
                    data=[_neurons(self.ann, self.labels, data=n_row,
                                   cmin=nv_min, cmax=nv_max),
                          _edges(self.ann, data=a_row,
                                 wmin=ew_min, wmax=ew_max,
                                 cmin=ev_min, cmax=ev_max)],
                    name=f'frame{index}'
                ) for index, (n_row, a_row)
                in enumerate(zip(self.neurons_data.data,
                                 self.axonal_data.data))
            ]
            fig = _figure(data=frames[0].data, frames=frames)

            def frame_args(duration):
                return {
                    "frame": {"duration": duration},
                    "mode": "immediate",
                    "fromcurrent": True,
                    "transition": {"duration": duration, "easing": "linear"},
                }

            if self.dt is None:
                def fmt(k): return str(k)
            else:
                def fmt(k): return f"{k*self.dt:g}s"

            sliders = [
                {"pad": {"b": 10, "t": 10},
                 "len": 0.9,
                 "x": 0.1,
                 "y": 0,

                 "steps": [
                     {"args": [[f.name], frame_args(0)],
                      "label": fmt(k),
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

                    "direction": "down",
                    "pad": {"r": 10, "t": 20, "b": 10},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }],
                sliders=sliders
            )

            self.axonal_data.to_csv(self.axonal_data_file)

            interactive_plot_file = self.axonal_data_file.with_suffix(".html")
            fig.write_html(interactive_plot_file,
                           auto_play=False)
            logger.info(f"Generated {interactive_plot_file}")


def _neurons(ann: ANN, labels: Dict[Point, str],
             data: Optional[Iterable[float]] = None,
             **kwargs) -> go.Scatter3d:
    # noinspection PyPep8Naming
    NT = ANN.Neuron.Type

    x, y, z = zip(*[n.pos.tuple() for n in _iter_neurons(ann)])
    names = []
    types = {NT.I: "Input", NT.H: "Hidden", NT.O: "Output"}
    n_types = {t: 0 for t in types}
    for n in _iter_neurons(ann):
        if labels is None or (label := labels.get(n.pos, None)) is None:
            label = f"{n.type.name}{n_types[n.type]}"
        n_types[n.type] += 1
        names.append(f"<b>{types[n.type]}</b><br>"
                     f"<i>{label}</i><br>"
                     f"Bias: {n.bias:.3g}")

    markers = dict(symbol='circle', size=10)

    custom_data = None
    hover_items = [
        "%{hovertext}", "x: %{x}", "y: %{y}", "z: %{z}"
    ]

    if data is None:
        color_dict = {NT.I: "black", NT.H: "grey", NT.O: "black"}
        markers['color'] = [color_dict[n.type] for n in _iter_neurons(ann)]
    else:
        markers.update(dict(
            color=data, showscale=True,
            colorscale=[
                "rgb(0, 0, 255)",
                "rgba(0, 0, 0, 0)",
                "rgb(255, 0, 0)"],
            cmin=kwargs.pop("cmin", None), cmax=kwargs.pop("cmax", None),
            colorbar=dict(
                len=0.5, y=.5,
                yanchor="top",
                title=dict(text="Neurons", side="right")
            )
        ))

        custom_data = markers['color']
        hover_items.append("value: %{customdata:.3}")

    return go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=markers,
            hovertemplate='<br>'.join(hover_items) + "<extra></extra>",
            customdata=custom_data,
            hovertext=names,
            **kwargs
        )


def _edges(ann: ANN, data: Optional[Iterable[float]] = None,
           **kwargs) -> go.Scatter3d:

    if ann.stats().edges == 0:
        return go.Scatter3d()

    x, y, z = (
        list(chain.from_iterable(lst)) for lst in
        zip(*[zip(*(src.pos.tuple(), dst.pos.tuple(), (None, None, None)))
              for _, src, dst in _iter_axons(ann)]))
    # w = [link.weight for link, _, _ in _iter_axons(ann)]

    c_min, c_max = kwargs.pop("cmin", None), kwargs.pop("cmax", None)
    _, _ = kwargs.pop("wmin", None), kwargs.pop("wmax", None)

    lines = dict(width=1)
    if data is None:
        lines['color'] = "black"
    else:
        # for (link, src, dst), v in zip(_iter_axons(ann), data):
        #     x0, y0, z0 = src.pos.tuple()
        #     x1, y1, z1 = dst.pos.tuple()
        #     print(f"({x0:+.3f}, {y0:+.3f}, {z0:+.3f})"
        #           f" -> ({x1:+.3f}, {y1:+.3f}, {z1:+.3f})"
        #           f" [w={link.weight:+.3f}, v={v:+.3f}]")
        data = list(chain.from_iterable([(x, x, x) for x in data]))
        lines.update(dict(
            color=data,
            showscale=True,
            colorscale=[
                [0, "rgba(0, 0, 255, 1)"],
                [.325, "rgba(0, 0, 0, 0)"],
                [.675, "rgba(0, 0, 0, 0)"],
                [1, "rgba(255, 0, 0, 1)"]],
            cmin=c_min, cmax=c_max,
            colorbar=dict(
                len=0.5, y=1,
                yanchor="top",
                title=dict(text="Axons", side="right")
            )
        ))

    return go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=lines,
        # opacity=.1,
        hoverinfo='none',
        **kwargs
    )


def _figure(**kwargs) -> go.Figure:

    fig = go.Figure(**kwargs)

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
        ),
        margin=dict(r=0, l=0, b=0, t=0)
    )

    return fig


class _TinyDataFrame:
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.data = []

    def append(self, data):
        self.data.append(data)

    def to_csv(self, file, sep=","):
        with open(file, "w") as f:
            f.write(sep.join(f"\"{s}\"" for s in [""] + self.columns) + "\n")
            for i, row in enumerate(self.data):
                f.write(sep.join(str(v) for v in [i] + row))
