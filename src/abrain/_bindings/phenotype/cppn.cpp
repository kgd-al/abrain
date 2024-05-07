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

template <uint DI>
void init_point_type (py::module &m, const char *name) {
  using Point = misc::Point_t<DI>;
  static constexpr auto Di = Point::DIMENSIONS;
  static constexpr auto De = Point::DECIMALS;

  auto pont = py::class_<Point>(m, name);

  pont.doc() = utils::mergeToString(
        Di, "D coordinate using fixed point notation with ", De, " decimals");
  pont.def("__repr__", [] (const Point &p) { return utils::mergeToString(p); })
      .def(pybind11::self == pybind11::self)
      .def(pybind11::self != pybind11::self)
      .def("__hash__", [] (const Point &p) { // Bad hash
        int sum = 0;
        for (const int i: p.data()) sum += i;
        return sum;
      })
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
  using Outputs = CPPN::Outputs;
  // using OutputSubset = typename CPPN::OutputSubset;

  auto cppn = py::class_<CPPN>(m, "CPPN");

  // using OutputsList = std::vector<Output>;
//  py::bind_vector<OutputsList>(cppn, "OutputsList");

  using Outputs = typename CPPN::Outputs;
  auto outp = py::class_<Outputs>(m, "Outputs");
  outp.doc() = "Output communication buffer for the CPPN";
  outp.def(py::init<>())
          .def("__len__", [] (const Outputs &o) { return o.size(); })
//      .def("__iter__", [] (Outputs &o) {
//        return py::make_iterator(o.begin(), o.end());
//      }, py::keep_alive<0, 1>())
          .def_property_readonly("__iter__", [] (const py::object&) { return py::none(); },
                                 "Cannot be iterated. Use direct access instead.")
          .def("__getitem__", [] (const Outputs &o, const size_t i) -> float { return o[i]; })
          ;

  // for (uint i=0; i<CPPN::OUTPUTS; i++)
  //   o_enum.value(cppn::CPPN_OUTPUT_ENUM_NAMES[i], CPPN::OUTPUTS_LIST[i]);

  // ** Duplicate **
  //  py::class_<OutputSubset>(m, "OutputSubset");

#define ID(X, ...) (#X, &CLASS::X, ##__VA_ARGS__)
#define CLASS CPPN
  cppn.def(py::init<const CPPN::Genotype&>())
      // .def_readonly_static ID(INPUTS)
      // .def_readonly_static ID(OUTPUTS)
      .def_static("outputs", [] { return Outputs(); },
                  "Return a buffer in which the CPPN can store output data")
      .def_static("functions", [] {
        using FFunction = std::function<float(float)>;
        std::map<typename CPPN::FuncID, FFunction> map;
        for (const auto &p: CPPN::functions)
          map[p.first] = FFunction(p.second);
        return map;
      }, "Return a copy of the C++ built-in function set")

      .def_readonly_static("_docstrings", &_cppn_docs)
    ;
  cppn.doc() = "Generic CPPN for regular use (images, morphologies, etc.)";
}

template <typename CPPN>
void init_eshn_cppn_phenotype (py::module_ &m, const char *name) {
  using Point = typename CPPN::Point;
  using Output = typename CPPN::Output;
  using Outputs = typename CPPN::Outputs;
  using OutputSubset = typename CPPN::OutputSubset;

  auto cppn = py::class_<CPPN>(m, name);
  auto o_enum = py::enum_<Output>(cppn, "Output");
  auto outp = py::class_<Outputs>(cppn, "Outputs");

  using OutputsList = std::vector<Output>;
//  py::bind_vector<OutputsList>(cppn, "OutputsList");

  static constexpr auto Di = Point::DIMENSIONS;
  static constexpr auto De = Point::DECIMALS;

#define ID(X, ...) (#X, &CLASS::X, ##__VA_ARGS__)

  // for (uint i=0; i<CPPN::OUTPUTS; i++)
  //   o_enum.value(cppn::CPPN_OUTPUT_ENUM_NAMES[i], CPPN::OUTPUTS_LIST[i]);

//  py::class_<OutputSubset>(m, "OutputSubset");

#define CLASS CPPN
  cppn.def(py::init<const typename CPPN::Genotype&>())
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

      // .def_readonly_static ID(DIMENSIONS)
      // .def_readonly_static ID(INPUTS)
      // .def_readonly_static ID(OUTPUTS)
      // .def_property_readonly_static("OUTPUTS_LIST", [] (py::object) {
      //   OutputsList values;
      //   for (uint i=0; i<CPPN::OUTPUTS; i++)
      //     values.push_back(CPPN::OUTPUTS_LIST[i]);
      //   return values;
      // })
      .def_static("outputs", [] { return Outputs(); },
                  "Return a buffer in which the CPPN can store output data")
      .def_static("functions", [] {
        using FFunction = std::function<float(float)>;
        std::map<typename CPPN::FuncID, FFunction> map;
        for (const auto &p: CPPN::functions)
          map[p.first] = FFunction(p.second);
        return map;
      }, "Return a copy of the C++ built-in function set")

      .def_readonly_static("_docstrings", &_cppn_docs)
    ;
  cppn.doc() = "Middle-man between the descriptive :class:`Genome` and the"
               " callable :class:`ANN`";
}
template void init_eshn_cppn_phenotype<CPPN2D>(py::module_ &m, const char *name);
template void init_eshn_cppn_phenotype<CPPN3D>(py::module_ &m, const char *name);

} // end of namespace kgd::eshn::pybind
