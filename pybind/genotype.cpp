#include <sstream>

#include "../cpp/genotype.h"

#include "pybind11/pybind11.h"
namespace py = pybind11;

#include "pybind11/stl_bind.h"
PYBIND11_MAKE_OPAQUE(std::vector<kgd::eshn::genotype::CPPNData::Node>)
PYBIND11_MAKE_OPAQUE(std::vector<kgd::eshn::genotype::CPPNData::Link>)

using namespace kgd::eshn::genotype;
namespace kgd::eshn::pybind {

void init_genotype (py::module_ &m) {
  using Node = CPPNData::Node;
  using Link = CPPNData::Link;

  auto cppn = py::class_<CPPNData>(m, "CPPNData");
  auto node = py::class_<Node>(cppn, "Node");
  auto link = py::class_<Link>(cppn, "Link");
  py::bind_vector<std::vector<Node>>(m, "Nodes");
  py::bind_vector<std::vector<Link>>(m, "Links");

#define ID(X) (#X, &CPPNData::X)
  cppn.def(py::init<>())
      .def_readonly_static ID(INPUTS)
      .def_readonly_static ID(OUTPUTS)
      .def_readwrite ID(nodes)
      .def_readwrite ID(links)
      .def_readwrite ID(nextNodeID)
      .def_readwrite ID(nextLinkID)
      ;

#undef ID
#define ID(X) (#X, &CPPNData::Node::X)
  node.def(py::init<int, const Node::FuncID &>())
      .def("__repr__", [] (const Node &n) {
        std::ostringstream oss;
        oss << "N" << n.id << ":" << n.func;
        return oss.str();
      })
      .def_readwrite ID(id)
      .def_readwrite ID(func);

#undef ID
#define ID(X) (#X, &CPPNData::Link::X)
  link.def(py::init<int, uint, uint, float>())
      .def("__repr__", [] (const Link &l) {
        std::ostringstream oss;
        oss << "L" << l.id << ":" << l.src << " -(" << l.weight << ")-> "
            << l.dst;
        return oss.str();
      })
      .def_readwrite ID(id)
      .def_readwrite ID(src)
      .def_readwrite ID(dst)
      .def_readwrite ID(weight);
}


} // end of namespace kgd::eshn::pybind
