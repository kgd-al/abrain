#include "../../cpp/phenotype/cppn.h"

#include "pybind11/pybind11.h"
namespace py = pybind11;

//#include "pybind11/stl_bind.h"

using namespace kgd::eshn::phenotype;
namespace kgd::eshn::pybind {

void init_cppn (py::module_ &m) {
  auto cppn = py::class_<CPPN>(m, "CPPN")
    ;
}

} // end of namespace kgd::eshn::pybind
