#include "../../_cpp/phenotype/cppn.h"

#include "pybind11/pybind11.h"
namespace py = pybind11;

#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

using namespace kgd::eshn::phenotype;
PYBIND11_MAKE_OPAQUE(CPPN::Outputs)

namespace kgd::eshn::pybind {

void init_cppn_phenotype (py::module_ &m) {
  using Point = CPPN::Point;
  using Output = CPPN::Genotype::Output;
  using Outputs = CPPN::Outputs;
  using OutputSubset = CPPN::OutputSubset;

  py::class_<Point>(m, "Point")
#if ESHN_SUBSTRATE_DIMENSION == 3
      .def(py::init([] (float x, float y, float z) { return Point({x,y,z}); }))
#endif
      .def("__repr__", [] (const Point &p) {
        std::ostringstream oss;
        oss << p;
        return oss.str();
      })
      .def("tuple", [] (const Point &p) {
#if ESHN_SUBSTRATE_DIMENSION == 3
        return std::tuple(p.x(), p.y(), p.z());
#else
        return std::tuple(p.x(), p.y());
#endif
      })
      ;

  auto o_enum = py::enum_<Output>(m, "Output");
  for (uint i=0; i<CPPN::OUTPUTS; i++)
    o_enum.value(cppn::CPPN_OUTPUT_ENUM_NAMES[i], CPPN::OUTPUTS_LIST[i]);

  py::class_<Outputs>(m, "Outputs")
      .def(py::init<>())
      .def("__len__", [] (const Outputs &o) { return o.size(); })
      .def("__iter__", [] (Outputs &o) {
        return py::make_iterator(o.begin(), o.end());
      }, py::keep_alive<0, 1>())
      .def("__getitem__", [] (Outputs &o, size_t i) { return o[i]; })
      ;
//  py::class_<OutputSubset>(m, "OutputSubset");

#define ID(X) (#X, &CPPN::X)
  auto cppn = py::class_<CPPN>(m, "CPPN")
      .def(py::init<const CPPN::Genotype&>())
      .def("__call__", py::overload_cast<const Point&,
                                         const Point&,
                                         Outputs&>(
                                           &CPPN::operator (), py::const_),
           "Evaluates on provided coordinates and retrieve all outputs")
      .def("__call__", py::overload_cast<const Point&,
                                         const Point&,
                                         Output>(
                                           &CPPN::operator (), py::const_),
           "Evaluates on provided coordinates for the requested output")
      .def("__call__", py::overload_cast<const Point&,
                                         const Point&,
                                         Outputs&,
                                         const OutputSubset&>(
                                           &CPPN::operator (), py::const_),
           "Evaluates on provided coordinates for the requested outputs")

      .def_readonly_static ID(DIMENSIONS)
      .def_readonly_static ID(INPUTS)
      .def_readonly_static ID(OUTPUTS)
      .def_property_readonly_static("OUTPUTS_LIST", [] (py::object) {
        std::vector<Output> values;
        for (uint i=0; i<CPPN::OUTPUTS; i++)
          values.push_back(CPPN::OUTPUTS_LIST[i]);
        return values;
      })
      .def_static("outputs", [] { return Outputs(); })
    ;

  py::bind_vector<std::vector<Output>>(m, "OutputList");
}

} // end of namespace kgd::eshn::pybind
