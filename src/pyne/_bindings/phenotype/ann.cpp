#include "../../_cpp/phenotype/ann.h"

#include "pybind11/pybind11.h"
namespace py = pybind11;

#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

using namespace kgd::eshn::phenotype;
PYBIND11_MAKE_OPAQUE(ANN::Inputs)
PYBIND11_MAKE_OPAQUE(ANN::NeuronsMap)
//PYBIND11_MAKE_OPAQUE(ANN::Neuron::Links)
//PYBIND11_MAKE_OPAQUE(ANN::Neuron::ptr)
// OPAQUE_TYPES?

namespace kgd::eshn::pybind {

void init_ann_phenotype (py::module_ &m) {
  using namespace pybind11::literals;
  using Neuron = ANN::Neuron;
  using Link = Neuron::Link;
  using Type = Neuron::Type;

  auto cann = py::class_<ANN>(m, "ANN");
  auto nern = py::class_<Neuron, std::shared_ptr<Neuron>>(cann, "Neuron");
  auto link = py::class_<Link>(nern, "Link");
  auto stts = py::class_<ANN::Stats>(cann, "Stats");

  auto nmap = py::class_<ANN::NeuronsMap>(cann, "Neurons");
  auto type = py::enum_<Type>(nern, "Type");
  py::bind_vector<ANN::Inputs>(cann, "IOValues");

#define ID(X, DOC) (#X, &ANN::X, DOC)
  cann.def(py::init<>())

      .def ID(inputs, "Return a ready-to-fill container for neural inputs")
      .def ID(outputs, "Return a ready-to-read container for neural outputs")
//      .def ID(reset)
      .def("__call__", &ANN::operator (),
           R"(Execute a computational step

Assigns provided input values to corresponding input neurons in the same order
as when created (see build). Returns output values as computed.
If not otherwise specified, a single computational substep is executed. If need
be (e.g. large network, fast response required) you can requested for multiple
sequential execution in one call

:param inputs: provided analog values for the input neurons
:param outputs: computed analog values for the output neurons
:param substeps: number of sequential executions

           )", // LCOV_EXCL_LINE
           "inputs"_a, "outputs"_a, "substeps"_a = 1)

      .def ID(empty,
              "Return whether or not the ANN contains neurons/connections")
      .def ID(stats, "Return associated stats (connections, depth...)")
      .def("neurons", py::overload_cast<>(&ANN::neurons, py::const_),
           "Provide read-only access to the underlying neurons")
      .def ID(neuronAt, "Query an individual neuron")

      .def_static("build", &ANN::build,
                  R"(Create an ANN via ES-HyperNEAT

The ANN has inputs/outputs at specified coordinates.
A CPPN is instantiated from the provided genome and used
to query connections weight, existence and to discover
hidden neurons locations

:param inputs: coordinates of the input neurons on the substrate
:param outputs: coordinates of the output neurons on the substrate
:param genome: genome describing a cppn (see :class:CPPN, :class:Genome)
                  )", "inputs"_a, "outputs"_a, "genome"_a)
  ;

  cann.doc() = "3D Artificial Neural Network produced through "
               "Evolvable Substrate Hyper-NEAT";

  nmap.def("__iter__", [] (ANN::NeuronsMap &m) {
        return py::make_iterator(m.begin(), m.end());
      }, py::keep_alive<0, 1>())
      .def("__len__", [] (const ANN::NeuronsMap &m) { return m.size(); })
      ;

#undef ID
#define ID(X) (#X, &Neuron::X)
  nern.def_readonly ID(pos)
      .def_readonly ID(type)
      .def_readonly ID(bias)
      .def_readonly ID(depth)
      .def_readonly ID(value)
      .def_readonly ID(flags)
      .def("links", py::overload_cast<>(&Neuron::links, py::const_))
  ;

#undef ID
#define ID(X) (#X, &Link::X)  
  link.def_readonly ID(weight)
      .def("src", [] (const Link &l){ return l.in.lock(); })
  ;

#undef ID
#define ID(X) (#X, Type::X)
  type.value ID(I)
      .value ID(H)
      .value ID(O)
  ;

#undef ID
#define ID(X) (#X, &ANN::Stats::X)
  stts.def_readonly ID(depth)
      .def_readonly ID(edges)
      .def_readonly ID(axons)
      ;
}

} // namespace kgd::eshn::pybind
