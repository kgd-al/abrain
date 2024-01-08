#include "../../_cpp/phenotype/ann.h"
#include "../utils.hpp"


#include <iostream>


#include "pybind11/pybind11.h"
namespace py = pybind11;

#include "pybind11/stl_bind.h"
using namespace pybind11::literals;

using namespace kgd::eshn::phenotype;
//PYBIND11_MAKE_OPAQUE(ANN::Inputs)
PYBIND11_MAKE_OPAQUE(ANN::NeuronsMap)
//PYBIND11_MAKE_OPAQUE(ANN::Neuron::Links)
//PYBIND11_MAKE_OPAQUE(ANN::Neuron::ptr)
// OPAQUE_TYPES?

namespace kgd::eshn::pybind {

#ifndef NDEBUG
py::tuple tuple(const std::vector<float> &v) { return py::tuple(py::cast(v)); }
void set_to_nan(std::vector<float> &v) { std::fill(v.begin(), v.end(), NAN); }
bool valid(const std::vector<float> &v) {
  return std::none_of(v.begin(), v.end(),
                      static_cast<bool(*)(float)>(std::isnan));
}

static constexpr auto doc_tuple = "Debug helper to convert to Python tuple";
static constexpr auto doc_set_to_nan = "Debug helper to set all values to NaN";
static constexpr auto doc_valid = "Debug tester to assert no values are NaN";
#endif

void init_ann_phenotype (py::module_ &m) {
  using IBuffer = ANN::IBuffer;
  using OBuffer = ANN::OBuffer;
  using Neuron = ANN::Neuron;
  using Link = Neuron::Link;
  using Type = Neuron::Type;

  auto cann = py::class_<ANN>(m, "ANN");
  auto nern = py::class_<Neuron, std::shared_ptr<Neuron>>(cann, "Neuron");
  auto link = py::class_<Link>(nern, "Link");
  auto stts = py::class_<ANN::Stats>(cann, "Stats");

  auto nmap = py::class_<ANN::NeuronsMap>(cann, "Neurons");
  auto type = py::enum_<Type>(nern, "Type");

  auto ibuf = py::class_<IBuffer, std::shared_ptr<IBuffer>>(cann, "IBuffer");
  auto obuf = py::class_<OBuffer, std::shared_ptr<OBuffer>>(cann, "OBuffer");
//  py::bind_vector<ANN::Inputs>(cann, "Buffer");

#define ID(X, ...) (#X, &CLASS::X, __VA_ARGS__)
#define CLASS ANN
  cann.def(py::init<>())

      .def ID(ibuffer, "Return a reference to the neural inputs buffer",
              py::return_value_policy::reference)
      .def ID(obuffer, "Return a reference to the neural outputs buffer",
              py::return_value_policy::reference)
      .def("buffers", [] (ANN &ann) -> std::tuple<IBuffer&, OBuffer&> {
        return std::tie(ann.ibuffer(), ann.obuffer());
      }, "Return the ann's I/O buffers as a tuple",
      py::return_value_policy::reference_internal)
//      .def ID(reset)
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

      .def ID(empty,
              R"(
Whether the ANN contains neurons/connections

:param strict: whether perceptrons count as empty (true) or not (false)

.. seealso:: `Config::allowPerceptrons`
              )", py::arg("strict") = false)
      .def ID(perceptron, "Whether this ANN is a perceptron")
      .def ID(stats, "Return associated stats (connections, depth...)")
      .def ID(reset, "Resets internal state to null (0)")
      .def("neurons", py::overload_cast<>(&ANN::neurons, py::const_),
           "Provide read-only access to the underlying neurons")
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

  cann.doc() = "3D Artificial Neural Network produced through "
               "Evolvable Substrate Hyper-NEAT";

  ibuf.doc() = "Specialized, fixed-size buffer for the neural inputs"
               " (write-only)";
  ibuf.def("__setitem__",
           [] (IBuffer &buf, size_t i, float v) { buf[i] = v; },
           "Assign an element")
      .def("__setitem__",
         [](IBuffer &buf, const py::slice &slice, const py::iterable &items) {
           size_t start = 0, stop = 0, step = 0, slicelength = 0;
           if (!slice.compute(buf.size(), &start, &stop, &step, &slicelength))
               throw py::error_already_set();
           if (slicelength != py::len(items))
               throw std::runtime_error("Left and right hand size of slice"
                                        " assignment have different sizes!");
           for (py::handle h: items) {
//           for (size_t i = 0; i < slicelength; ++i) {
               buf[start] = h.cast<float>();
               start += step;
           }
       })
      .def("__len__", [] (const ANN::IBuffer &buf) { return buf.size(); },
           "Return the number of expected inputs")
#ifndef NDEBUG
      .def("tuple", [] (const IBuffer &buf) { return tuple(buf); }, doc_tuple)
      .def("set_to_nan", [] (IBuffer &buf) { return set_to_nan(buf); }, doc_set_to_nan)
      .def("valid", [] (const IBuffer &buf) { return valid(buf); }, doc_valid)
#endif
  ;

  obuf.doc() = "Specialized, fixed-size buffer for the neural outputs"
               " (read-only)";
  obuf.def("__getitem__",
           [] (const OBuffer &buf, size_t i) { return buf[i]; },
           "Access an element")
      .def("__getitem__",
        [](const OBuffer &buf, const py::slice &slice) -> py::list * {
        size_t start = 0, stop = 0, step = 0, slicelength = 0;
        if (!slice.compute(buf.size(), &start, &stop, &step, &slicelength))
           throw py::error_already_set();

        auto *list = new py::list(slicelength);
        for (size_t i = 0; i < slicelength; ++i) {
           (*list)[i] = buf[start];
           start += step;
        }
        return list;
      })
      .def("__len__", [] (const OBuffer &buf) { return buf.size(); },
           "Return the number of expected outputs")
      .def_property_readonly("__iter__", [] (py::object) { return py::none(); },
           "Cannot be iterated. Use direct access instead.")
#ifndef NDEBUG
      .def("tuple", [] (const OBuffer &buf) { return tuple(buf); }, doc_tuple)
      .def("set_to_nan", [] (OBuffer &buf) { set_to_nan(buf); }, doc_set_to_nan)
      .def("valid", [] (const OBuffer &buf) { return valid(buf); }, doc_valid)
#endif
      ;

  nmap.doc() = "Wrapper for the C++ neurons container";
  nmap.def("__iter__", [] (ANN::NeuronsMap &m) {
        return py::make_iterator(m.begin(), m.end());
      }, py::keep_alive<0, 1>())
      .def("__len__", [] (const ANN::NeuronsMap &m) { return m.size(); })
      ;

#undef CLASS
#define CLASS Neuron
  nern.doc() = "Atomic computational unit of an ANN";
  nern.def_readonly ID(pos, utils::mergeToString(
                         "Position in the ", ESHN_SUBSTRATE_DIMENSION,
                         "D substrate").c_str())
      .def_readonly ID(type, "Neuron role (see :class:`Type`)")
      .def_readonly ID(bias, "Neural bias")
      .def_readonly ID(depth, "Depth in the neural network")
      .def_readonly ID(value, "Current activation value")
      .def_readonly ID(flags, "Stimuli-dependent flags (for modularization)")
      .def("links", py::overload_cast<>(&Neuron::links, py::const_),
           "Return the list of inputs connections")
  ;

#undef CLASS
#define CLASS Link
  link.doc() = "An incoming neural connection";
  link.def_readonly ID(weight, "Connection weight "
                               "(see attr:`Config.annWeightScale`)")
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
      .def_readonly ID(iterations, "H -> H iterations before convergence")
      .def("dict", [] (const CLASS &stats) {
        return py::dict (
#define PAIR(X) #X##_a=stats.X
          PAIR(depth), PAIR(iterations),
          PAIR(hidden),
          PAIR(edges), PAIR(axons),
          PAIR(density)
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

} // namespace kgd::eshn::pybind
