#include "../../_cpp/phenotype/cppn.h"
#include "../utils.hpp"

#include <functional>
#include <iostream>

#include "pybind11/pybind11.h"
namespace py = pybind11;

#include "pybind11/functional.h"
#include "pybind11/operators.h"
using namespace pybind11::literals;

using namespace kgd::eshn::phenotype;
PYBIND11_MAKE_OPAQUE(CPPN::IBuffer)
PYBIND11_MAKE_OPAQUE(CPPN::OBuffer)

namespace kgd::eshn::pybind {

#define ID(X, ...) (#X, &CLASS::X, __VA_ARGS__)

template <typename py_type>
CPPN::IBuffer fromList (const py_type &iterable) {
  CPPN::IBuffer buffer;
  for (const auto &v: iterable) buffer.push_back(v.template cast<float>());
  return buffer;
}

static const utils::DocMap _cppn_docs {
  { "DIMENSIONS", "for the I/O coordinates" },
  { "INPUTS", "Number of inputs" },
  { "OUTPUTS", "Number of outputs" },
  { "OUTPUTS_LIST", "The list of output types the CPPN can produce" }
};

template <unsigned int DI>
void init_point_type (py::module &m, const char *name) {
  using Point = misc::Point_t<DI>;
  static constexpr auto Di = Point::DIMENSIONS;
  static constexpr auto De = Point::DECIMALS;

  auto pont = py::class_<Point>(m, name);

  pont.doc() = utils::mergeToString(
        Di, "D coordinate using fixed point notation with ", De, " decimals");
  pont.def("__repr__", [] (const Point &p) {
        return utils::mergeToString("Point", Di, "D(", p, ")");
       })
      .def(pybind11::self == pybind11::self)
      .def(pybind11::self != pybind11::self)
      .def("__hash__", [] (const Point &p) { // Bad hash
        int sum = 0;
        for (const int i: p.data()) sum += i;
        return sum;
      })
      .def("null", Point::null, "Return the null vector")
      .def_readonly_static("DIMENSIONS", &Point::DIMENSIONS)
      ;

  if constexpr (Di == 2) {
    pont.def(
        py::init([] (float x, float y) { return Point({x,y}); }),
            "Create a point with the specified coordinates\n\n"
            "Args:\n"
            "  x, y (float): x, y coordinate",
            "x"_a, "y"_a)
        .def("tuple", [] (const Point &p) {
          return std::tuple(p.x(), p.y());
        }, "Return a tuple for easy unpacking in python");
  } else {
    pont.def(
        py::init([] (float x, float y, float z) { return Point({x,y,z}); }),
            "Create a point with the specified coordinates\n\n"
            "Args:\n"
            "  x, y, z (float): x, y, z coordinate",
            "x"_a, "y"_a, "z"_a)
        .def("tuple", [] (const Point &p) {
          return std::tuple(p.x(), p.y(), p.z());
        }, "Return a tuple for easy unpacking in python");
  }
}
template void init_point_type<2>(py::module_ &m, const char *name);
template void init_point_type<3>(py::module_ &m, const char *name);

void init_generic_cppn_phenotype (py::module_ &m) {
  auto cppn = py::class_<CPPN>(m, "CPPN");

  using IBuffer = typename CPPN::IBuffer;
  using OBuffer = typename CPPN::OBuffer;

  utils::init_buffer<IBuffer>(cppn, "IBuffer", "Input data buffer for a CPPN");
  utils::init_buffer<OBuffer>(cppn, "OBuffer", "Output data buffer for a CPPN");

#define CLASS CPPN
  cppn.def(py::init<const CPPN::Genotype&>())
      .def ID(n_inputs, "Return the number of inputs", py::arg("with_bias") = false)
      .def ID(n_outputs, "Return the number of outputs")
      .def ID(n_hidden, "Return the number of internal nodes")

      .def ID(ibuffer, "Buffer for input data")
      .def ID(obuffer, "Buffer for output data")

      .def("outputs", &CPPN::obuffer,
           "Return a buffer in which the CPPN can store output data")
      .def_static("functions", [] {
          using FFunction = std::function<float(float)>;
          std::map<typename CPPN::FuncID, FFunction> map;
          for (const auto &p: CPPN::functions)
            map[p.first] = FFunction(p.second);
          return map;
      }, "Return a copy of the C++ built-in function set")

      .def("__call__", py::overload_cast<OBuffer&, const IBuffer&>(
                   &CPPN::operator ()),
           "Evaluates on provided inputs and retrieve all outputs",
           "outputs"_a, "inputs"_a)
      .def("__call__", py::overload_cast<unsigned int, const IBuffer&>(
                   &CPPN::operator ()),
           "Evaluates on provided inputs and retrieve requested output",
           "output"_a, "inputs"_a)

      .def("__call__",
           [] (CPPN &cppn, OBuffer &o, const py::list &inputs) {
               cppn(o, fromList(inputs));
           },
           "Evaluates on provided inputs and retrieve all outputs",
           "outputs"_a, "inputs"_a)
      .def("__call__",
           [] (CPPN &cppn, unsigned int o, const py::list &inputs) {
               return cppn(o, fromList(inputs));
           },
           "Evaluates on provided inputs and retrieve requested output",
           "output"_a, "inputs"_a)

      .def("__call__",
           [] (CPPN &cppn, OBuffer &o, const py::args &inputs) {
               cppn(o, fromList(inputs));
           },
           "Evaluates on provided inputs and retrieve all outputs",
           "outputs"_a)
      .def("__call__",
           [] (CPPN &cppn, unsigned int o, const py::args &inputs) {
               return cppn(o, fromList(inputs));
           },
           "Evaluates on provided inputs and retrieve requested output",
           "output"_a)

      .def_readonly_static("_docstrings", &_cppn_docs)
    ;
  cppn.doc() = "Generic CPPN for regular use (images, morphologies, etc.)";
}

template <typename CPPN>
void init_eshn_cppn_phenotype (py::module_ &m, const char *name, const char *pname) {
  using Point = typename CPPN::Point;
  using Output = typename CPPN::Output;
  using OBuffer = typename CPPN::OBuffer;
  using OutputSubset = typename CPPN::OutputSubset;

  auto cppn = py::class_<CPPN, phenotype::CPPN>(m, name);

#define CLASS CPPN
  cppn.def(py::init<const typename CPPN::Genotype&>(),
           "Create from a :class:`abrain.Genome`")
      .def_readonly_static
          ID(DIMENSIONS,
             "Dimensions of the points provided as inputs")
      .def("__call__",
           py::overload_cast<const Point&, const Point&, OBuffer&>(
                   &CPPN::operator ()),
           "Evaluates on provided coordinates and retrieve all outputs",
           "src"_a, "dst"_a, "buffer"_a)
      .def("__call__",
           py::overload_cast<const Point&, const Point&, Output>(
                   &CPPN::operator ()),
           "Evaluates on provided coordinates for the requested output\n\n"
           ".. note: due to an i686 bug this function is unoptimized on said"
           " platforms",
           "src"_a, "dst"_a, "type"_a)
      .def("__call__",
           py::overload_cast<const Point&, const Point&, OBuffer&, const OutputSubset&>(
                   &CPPN::operator ()),
           "Evaluates on provided coordinates for the requested outputs",
           "src"_a, "dst"_a, "buffer"_a, "subset"_a)
      ;
  cppn.attr("Point") = m.attr(pname);
  cppn.doc() = utils::mergeToString(
          "Created from a :class:`~abrain.Genome` and used to generate,"
          " via ES-HyperNEAT, an :class:`~abrain.ANN", CPPN::DIMENSIONS, "D`");
}
template void init_eshn_cppn_phenotype<CPPN2D>(py::module_ &m, const char *name, const char *pname);
template void init_eshn_cppn_phenotype<CPPN3D>(py::module_ &m, const char *name, const char *pname);

} // end of namespace kgd::eshn::pybind
