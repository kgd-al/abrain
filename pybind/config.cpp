#include "../cpp/config.h"

#include <stdio.h>

#include "pybind11/pybind11.h"
namespace py = pybind11;

#include "pybind11/stl_bind.h"
using namespace kgd::eshn;

PYBIND11_MAKE_OPAQUE(Config::Functions)
PYBIND11_MAKE_OPAQUE(Config::MutationRates)

namespace kgd::eshn::pybind {

#define ID(X) (#X, &CLASS::X)
void init_config (py::module_ &m) {
#define CLASS Config
  auto config = py::class_<Config>(m, "Config")
      .def_readwrite_static ID(functionSet)
      .def_readwrite_static ID(outputFunctions)
      .def_readwrite_static ID(mutationRates)
      .def_readwrite_static ID(weightBounds)
      .def_property_readonly_static("cppnInputNames", [] (py::object){
        std::vector<std::string> names;
        for (uint i=0; i<cppn::CPPN_INPUTS; i++)
          names.push_back(std::string(cppn::CPPN_INPUT_NAMES[i]));
        return names;
      })
      .def_property_readonly_static("cppnOutputNames", [] (py::object){
        std::vector<std::string> names;
        for (uint i=0; i<cppn::CPPN_OUTPUTS; i++)
          names.push_back(std::string(cppn::CPPN_OUTPUT_NAMES[i]));
        return names;
      });

#undef CLASS
#define CLASS Config::FBounds
  py::class_<Config::FBounds>(m, "FBounds")
      .def_readwrite ID(rndMin).def_readwrite ID(min)
      .def_readwrite ID(rndMax).def_readwrite ID(max);

  py::bind_vector<Config::Functions>(m, "Functions");
  py::bind_map<Config::MutationRates>(m, "MutationRates");
//  py::bind_vector<Config::FunctionSet>(m, "FunctionSet");
}

} // end of namespace kgd::eshn::pybind
