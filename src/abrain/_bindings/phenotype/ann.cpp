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

namespace kgd::eshn::pybind {

template <typename ANN>
void init_ann_phenotype (py::module_ &m, const char *name) {
  // std::cerr << "\n== Registering " << m.attr("__name__").template cast<std::string>() << " ==\n"<< std::endl;

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
//      .def("neurons", py::overload_cast<>(&ANN::neurons, py::const_),
//           "Provide read-only access to the underlying neurons")
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
      .def("dict", [] (const CLASS &stats) {
        return py::dict (
#define PAIR(X) #X##_a=stats.X
          PAIR(depth), PAIR(iterations),
          PAIR(hidden),
          PAIR(edges), PAIR(axons),
          PAIR(density), PAIR(utility)
#undef PAIR
#ifndef NDEBUG
          , "time"_a=py::dict(
            "build"_a=stats.time.build,
            "eval"_a=stats.time.eval
          )
#endif
        );
      }, "Return the stats as Python dictionary")
      ;
}

template void init_ann_phenotype<ANN2D> (py::module_ &m, const char *name);
template void init_ann_phenotype<ANN3D> (py::module_ &m, const char *name);


} // namespace kgd::eshn::pybind
