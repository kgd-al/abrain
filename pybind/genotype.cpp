#include "../cpp/genotype.h"

#include "pybind11/pybind11.h"
namespace py = pybind11;

#include "pybind11/stl_bind.h"
PYBIND11_MAKE_OPAQUE(std::vector<kgd::eshn::genotype::CPPNData::Node>)
PYBIND11_MAKE_OPAQUE(std::vector<kgd::eshn::genotype::CPPNData::Link>)

using namespace kgd::eshn::genotype;
namespace kgd::eshn::pybind {

void init_genotype (py::module_ &m) {

#define ID(X) (#X, &CPPNData::X)
  auto cppn = py::class_<CPPNData>(m, "CPPNData")
      .def(py::init<>())
      .def_readonly_static ID(INPUTS)
      .def_readonly_static ID(OUTPUTS)
      .def_readwrite ID(nodes)
      .def_readwrite ID(links);

#undef ID
#define ID(X) (#X, &CPPNData::Node::X)
  py::class_<CPPNData::Node>(cppn, "Node")
      .def(py::init<const CPPNData::Node::FuncID &>())
      .def_readwrite ID(func);

#undef ID
#define ID(X) (#X, &CPPNData::Link::X)
  py::class_<CPPNData::Link>(cppn, "Link")
      .def(py::init<uint, uint, float>())
      .def_readwrite ID(src)
      .def_readwrite ID(dst)
      .def_readwrite ID(weight);

  py::bind_vector<std::vector<CPPNData::Node>>(m, "Nodes");
  py::bind_vector<std::vector<CPPNData::Link>>(m, "Links");
}


} // end of namespace kgd::eshn::pybind
