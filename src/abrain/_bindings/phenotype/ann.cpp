#include "../../_cpp/phenotype/ann.h"
#include "../utils.hpp"

#include "pybind11/pybind11.h"
namespace py = pybind11;

#include "pybind11/stl_bind.h"
using namespace pybind11::literals;

using namespace kgd::eshn::phenotype;
//PYBIND11_MAKE_OPAQUE(ANN::Inputs)
PYBIND11_MAKE_OPAQUE(ANN2D::NeuronsMap)
PYBIND11_MAKE_OPAQUE(ANN3D::NeuronsMap)
//PYBIND11_MAKE_OPAQUE(ANN::Neuron::Links)
//PYBIND11_MAKE_OPAQUE(ANN::Neuron::ptr)
// OPAQUE_TYPES?

namespace kgd::eshn::phenotype {

template <unsigned int DIMENSIONS>
struct ANNSerializer {
  using ANN = ANN_t<DIMENSIONS>;
  using Neuron = typename ANN::Neuron;
  using Point = typename ANN::Point;
  using Stats = typename ANN::Stats;

  static py::dict stats_to_json(const Stats &s) {
    return py::dict (
#define PAIR(X) #X##_a=s.X
      PAIR(depth), PAIR(iterations),
      PAIR(hidden),
      PAIR(edges), PAIR(axons),
      PAIR(density), PAIR(utility)
#undef PAIR
#ifndef NDEBUG
      ,
      "time"_a=py::dict(
        "build"_a=s.time.build,
        "eval"_a=s.time.eval
      )
#endif
    );
  }

  static Stats stats_from_json(const py::dict &d) {
    Stats s;
#define GET(X) s.X = d[#X].cast<decltype(s.X)>()
    GET(depth); GET(iterations);
    GET(hidden);
    GET(edges); GET(axons);
    GET(density); GET(utility);
#undef GET
#ifndef NDEBUG
    s.time.build = d["time"]["build"].cast<typename Stats::rep>();
    s.time.eval = d["time"]["eval"].cast<typename Stats::rep>();
#endif
    return s;
  }

  static py::dict to_json(const ANN &ann) {
    py::dict dict;

    py::list neurons, links;

    for (const typename ANN::NeuronPtr &np: ann._neurons) {
      const auto &n = *np;
      neurons.append(py::make_tuple(
        n.pos, n.type, n.bias, n.value, n.depth, n.flags));
    }

    for (const typename ANN::NeuronPtr &np: ann._neurons) {
      const Neuron &to = *np;
      const auto to_id = to.pos;
      for (const typename ANN::Neuron::Link &l: to.links()) {
        links.append(py::make_tuple(
          l.in.lock()->pos, to_id, l.weight
        ));
      }
    }

    dict["neurons"] = neurons;
    dict["links"] = links;
    dict["stats"] = stats_to_json(ann.stats());

    return dict;
  }

  static ANN from_json(const py::dict &d) {
    ANN ann;
    for (const py::handle &h: d["neurons"]) {
      const auto &t = h.cast<py::tuple>();
      auto n = ann.addNeuron(
          t[0].cast<Point>(),
          t[1].cast<typename Neuron::Type>(),
          t[2].cast<float>());
      n->value = t[3].cast<float>();
      n->depth = t[4].cast<unsigned int>();
      n->flags = t[5].cast<typename Neuron::Flags_t>();
      ann._neurons.insert(n);

      switch (n->type) {
      case Neuron::Type::I:
        ann._inputs.push_back(n);
        break;
      case Neuron::Type::O:
        ann._outputs.push_back(n);
        break;
      default:
        break;
      }
    }

    ann._ibuffer.resize(ann._inputs.size());
    ann._obuffer.resize(ann._outputs.size());

    for (const py::handle &h: d["links"]) {
      const auto &t = h.cast<py::tuple>();
      auto from_id = t[0].cast<Point>();
      auto to_id = t[1].cast<Point>();
      auto weight = t[2].cast<float>();

      ann.neuronAt(to_id)->addLink(weight, ann.neuronAt(from_id));
    }

    ann._stats = stats_from_json(d["stats"]);

    return ann;
  }
};

}

namespace kgd::eshn::pybind {

template <typename ANN>
void init_ann_phenotype (py::module_ &m, const char *name) {
  // std::cerr << "\n== Registering " << m.attr("__name__").template cast<std::string>() << " ==\n"<< std::endl;

  using Serializer = ANNSerializer<ANN::DIMENSIONS>;

  using Point = typename ANN::Point;
  static constexpr auto D = Point::DIMENSIONS;

  using IBuffer = typename ANN::IBuffer;
  using OBuffer = typename ANN::OBuffer;
  using Neuron = typename ANN::Neuron;
  using Link = typename Neuron::Link;
  using Type = typename Neuron::Type;

  auto cann = py::class_<ANN>(m, name);
  auto nern = py::class_<Neuron, std::shared_ptr<Neuron>>(cann, "Neuron");
  auto link = py::class_<Link>(nern, "Link");

  using Stats = typename ANN::Stats;
  auto stts = py::class_<Stats>(cann, "Stats");

  using NeuronsMap = typename ANN::NeuronsMap;
  auto nmap = py::class_<NeuronsMap>(cann, "Neurons");
  auto type = py::enum_<Type>(nern, "Type");

  utils::init_buffer<IBuffer>(cann, "IBuffer", "Input data buffer for an ANN");
  utils::init_buffer<OBuffer>(cann, "OBuffer", "Output data buffer for an ANN");

#define ID(X, ...) (#X, &CLASS::X, ##__VA_ARGS__)
#define CLASS ANN
  cann.def ID(ibuffer, "Return a reference to the neural inputs buffer",
              py::return_value_policy::reference)
      .def ID(obuffer, "Return a reference to the neural outputs buffer",
              py::return_value_policy::reference)
      .def("buffers", [] (ANN &ann) -> std::tuple<IBuffer&, OBuffer&> {
        return std::tie(ann.ibuffer(), ann.obuffer());
      }, "Return the ann's I/O buffers as a tuple",
      py::return_value_policy::reference_internal)
      .def("__call__", &ANN::operator (), R"(
Execute a computational step

Assigns provided input values to corresponding input neurons in the same order
as when created (see build). Returns output values as computed.
If not otherwise specified, a single computational substep is executed. If need
be (e.g. large network, fast response required) you can requested for multiple
sequential execution in one call

:param inputs: provided analog values for the input neurons
:param outputs: computed analog values for the output neurons
:param substeps: number of sequential executions

.. seealso:: :ref:`usage-basics-ann`
           )", "inputs"_a, "outputs"_a, "substeps"_a = 1)

      .def_readonly_static ID(DIMENSIONS,
                              "Specifies the underlying substrate dimension")

      .def ID(empty,
              R"(
Whether the ANN contains neurons/connections

:param strict: whether perceptrons count as empty (true) or not (false)

.. seealso:: `Config::allowPerceptrons`
              )", py::arg("strict") = false)
      .def ID(perceptron, "Whether this ANN is a perceptron")
      .def ID(max_hidden_neurons, "How many hidden neurons an ANN could have"
                                  " based on the value of"
                                  " :attr:`~abrain.Config.maxDepth`")
      .def ID(max_edges, "How many connections this ANN could have based on "
                         " the number of inputs/outputs and hidden nodes"
                         " (if any)")
      .def ID(stats, "Return associated stats (connections, depth...)")
      .def ID(reset, "Resets internal state to null (0)")
      .def("neurons", static_cast<const NeuronsMap& (ANN::*)() const>(&ANN::neurons),
           "Provide read-only access to the underlying neurons")
      .def("input_neurons", &ANN::input_neurons,
           "Provide read-only access to the underlying input neurons, in the same order as they"
           " are filled in with data")
      .def("output_neurons", &ANN::output_neurons,
           "Provide read-only access to the underlying output neurons, in the same order as they"
           " are read out for data")
      .def ID(neuronAt, "Query an individual neuron", "pos"_a)

      .def_static("build", &ANN::build, R"(
Create an ANN via ES-HyperNEAT

The ANN has inputs/outputs at specified coordinates.
A CPPN is instantiated from the provided genome and used
to query connections weight, existence and to discover
hidden neurons locations

:param inputs: coordinates of the input neurons on the substrate
:param outputs: coordinates of the output neurons on the substrate
:param genome: genome describing a cppn (see :class:`abrain.Genome`,
                                        :class:`CPPN`)

.. seealso:: :ref:`usage-basics-ann`
                  )", "inputs"_a, "outputs"_a, "genome"_a)

      .def("to_json", Serializer::to_json, "Convert to a json-compliant Python dictionary")
      .def_static("from_json", Serializer::from_json, "j"_a,
                "Convert from the json-compliant Python dictionary `j`")
      .def(py::pickle(
        [](const CLASS &ann) { return Serializer::to_json(ann); },
        [](const py::dict &d) { return Serializer::from_json(d);  }
      ))
      ;

  cann.doc() = utils::mergeToString(
          D,
          "D Artificial Neural Network produced through "
          "Evolvable Substrate Hyper-NEAT");

  nmap.doc() = "Wrapper for the C++ neurons container";
  nmap.def("__iter__", [] (NeuronsMap &_m) {
        return py::make_iterator(_m.begin(), _m.end());
      }, py::keep_alive<0, 1>())
      .def("__len__", [] (const NeuronsMap &_m) { return _m.size(); })
      ;

#undef CLASS
#define CLASS Neuron
  nern.doc() = "Atomic computational unit of an ANN";
  nern.def_readonly ID(pos, utils::mergeToString(
                         "Position in the ", ANN::Point::DIMENSIONS,
                         "D substrate").c_str())
      .def_readonly ID(type, "Neuron role (see :class:`Type`)")
      .def_readonly ID(bias, "Neural bias")
      .def_readonly ID(depth, "Depth in the neural network")
      .def_readonly ID(value, "Current activation value")
      .def_readonly ID(flags, "Stimuli-dependent flags (for modularization)")
      .def("links", py::overload_cast<>(&Neuron::links, py::const_),
           "Return the list of inputs connections")
      .def("is_input", &Neuron::isInput,
           "Whether this neuron is used a an input")
      .def("is_output", &Neuron::isOutput,
           "Whether this neuron is used a an output")
      .def("is_hidden", &Neuron::isHidden,
           "Whether this neuron is used for internal computations")
  ;

#undef CLASS
#define CLASS Link
  link.doc() = "An incoming neural connection";
  link.def_readonly ID(weight, "Connection weight "
                               "(see :attr:`abrain.Config.annWeightsRange`)")
      .def("src", [] (const Link &l){ return l.in.lock(); },
           "Return a reference to the source neuron")
  ;

#undef CLASS
#define CLASS Type
  type.value("I", Type::I, "Input (receiving data)")
      .value("H", Type::H, "Hidden (processing data)")
      .value("O", Type::O, "Output (producing data)")
  ;

#undef CLASS
#define CLASS ANN::Stats
  stts.doc() = "Contains various statistics about an ANN";
  stts.def_readonly ID(hidden, "Number of hidden neurons")
      .def_readonly ID(depth, "Maximal depth of the neural network")
      .def_readonly ID(edges, "Number of connections")
      .def_readonly ID(axons, "Total length of the connections")
      .def_readonly ID(density, "Ratio of expressed connections")
      .def_readonly ID(utility, "Ratio of used input/output neurons")
      .def_readonly ID(iterations, "H -> H iterations before convergence")
      .def("dict", Serializer::stats_to_json, "Return the stats as Python dictionary")
      ;
}

template void init_ann_phenotype<ANN2D> (py::module_ &m, const char *name);
template void init_ann_phenotype<ANN3D> (py::module_ &m, const char *name);


} // namespace kgd::eshn::pybind
