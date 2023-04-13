#include "../../_cpp/phenotype/cppn.h"
#include "../utils.hpp"

#include <functional>

#include "pybind11/pybind11.h"
namespace py = pybind11;

#include "pybind11/stl_bind.h"
#include "pybind11/functional.h"
#include "pybind11/operators.h"
using namespace pybind11::literals;

using namespace kgd::eshn::phenotype;
PYBIND11_MAKE_OPAQUE(CPPN::Outputs)

namespace kgd::eshn::pybind {

static const utils::DocMap _cppn_docs {
  { "DIMENSIONS", "for the I/O coordinates" },
  { "INPUTS", "Number of inputs" },
  { "OUTPUTS", "Number of outputs" },
  { "OUTPUTS_LIST", "The list of output types the CPPN can produce" }
};

void init_cppn_phenotype (py::module_ &m) {
  using Point = CPPN::Point;
  using Output = CPPN::Genotype::Output;
  using Outputs = CPPN::Outputs;
  using OutputSubset = CPPN::OutputSubset;

  auto cppn = py::class_<CPPN>(m, "CPPN");
  auto pont = py::class_<Point>(m, "Point");
  auto o_enum = py::enum_<Output>(cppn, "Output");
  auto outp = py::class_<Outputs>(cppn, "Outputs");

  using OutputsList = std::vector<Output>;
//  py::bind_vector<OutputsList>(cppn, "OutputsList");

  static constexpr auto Di = Point::DIMENSIONS;
  static constexpr auto De = Point::DECIMALS;

#define ID(X, ...) (#X, &CLASS::X, ##__VA_ARGS__)

  pont.doc() = utils::mergeToString(
        Di, "D coordinate using fixed point notation with ", De, " decimals");
  pont
#if ESHN_SUBSTRATE_DIMENSION == 3
      .def(py::init([] (float x, float y, float z) { return Point({x,y,z}); }),
           "Create a point with the specified coordinates\n\n"
           "Args:\n"
           "  x, y, z (float): x, y, z coordinate",
           "x"_a, "y"_a, "z"_a)
#endif
      .def("__repr__", [] (const Point &p) { return utils::mergeToString(p); })
      .def(pybind11::self == pybind11::self)
      .def(pybind11::self != pybind11::self)
      .def("__hash__", [] (const Point &p) { // Bad hash
        int sum = 0;
        for (int i: p.data()) sum += i;
        return sum;
      })
      .def("tuple", [] (const Point &p) {
#if ESHN_SUBSTRATE_DIMENSION == 3
        return std::tuple(p.x(), p.y(), p.z());
#else
        return std::tuple(p.x(), p.y());
#endif
      }, "Return a tuple for easy unpacking in python")
      ;

  for (uint i=0; i<CPPN::OUTPUTS; i++)
    o_enum.value(cppn::CPPN_OUTPUT_ENUM_NAMES[i], CPPN::OUTPUTS_LIST[i]);

  outp.doc() = "Output communication buffer for the CPPN";
  outp.def(py::init<>())
      .def("__len__", [] (const Outputs &o) { return o.size(); })
//      .def("__iter__", [] (Outputs &o) {
//        return py::make_iterator(o.begin(), o.end());
//      }, py::keep_alive<0, 1>())
      .def_property_readonly("__iter__", [] (py::object) { return py::none(); },
           "Cannot be iterated. Use direct access instead.")
      .def("__getitem__", [] (Outputs &o, size_t i) -> float { return o[i]; })
      ;
//  py::class_<OutputSubset>(m, "OutputSubset");

#define CLASS CPPN
  cppn.def(py::init<const CPPN::Genotype&>())
      .def("__call__", py::overload_cast<const Point&,
                                         const Point&,
                                         Outputs&>(
                                           &CPPN::operator ()),
           "Evaluates on provided coordinates and retrieve all outputs",
           "src"_a, "dst"_a, "buffer"_a)
      .def("__call__", py::overload_cast<const Point&,
                                         const Point&,
                                         Output>(
                                           &CPPN::operator ()),
           "Evaluates on provided coordinates for the requested output\n\n"
           ".. note: due to an i686 bug this function is unoptimized on said"
           " platforms",
           "src"_a, "dst"_a, "type"_a)
      .def("__call__", py::overload_cast<const Point&,
                                         const Point&,
                                         Outputs&,
                                         const OutputSubset&>(
                                           &CPPN::operator ()),
           "Evaluates on provided coordinates for the requested outputs",
           "src"_a, "dst"_a, "buffer"_a, "subset"_a)

      .def_readonly_static ID(DIMENSIONS)
      .def_readonly_static ID(INPUTS)
      .def_readonly_static ID(OUTPUTS)
      .def_property_readonly_static("OUTPUTS_LIST", [] (py::object) {
        OutputsList values;
        for (uint i=0; i<CPPN::OUTPUTS; i++)
          values.push_back(CPPN::OUTPUTS_LIST[i]);
        return values;
      })
      .def_static("outputs", [] { return Outputs(); },
                  "Return a buffer in which the CPPN can store output data")
      .def_static("functions", [] {
        using FFunction = std::function<float(float)>;
        std::map<CPPN::FuncID, FFunction> map;
        for (const auto &p: CPPN::functions)
          map[p.first] = FFunction(p.second);
        return map;
      }, "Return a copy of the C++ built-in function set")

      .def_readonly_static("_docstrings", &_cppn_docs)
    ;
  cppn.doc() = "Middle-man between the descriptive :class:`Genome` and the"
               " callable :class:`ANN`";
}

} // end of namespace kgd::eshn::pybind
