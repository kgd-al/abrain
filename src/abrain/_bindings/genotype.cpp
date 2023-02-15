#include <sstream>

#include "../_cpp/genotype.h"
#include "utils.hpp"

#include "pybind11/pybind11.h"
namespace py = pybind11;
using namespace pybind11::literals;

#include "pybind11/stl_bind.h"
PYBIND11_MAKE_OPAQUE(std::vector<kgd::eshn::genotype::CPPNData::Node>)
PYBIND11_MAKE_OPAQUE(std::vector<kgd::eshn::genotype::CPPNData::Link>)

using namespace kgd::eshn::genotype;
namespace kgd::eshn::pybind {

static const utils::DocMap _cppn_doc {
  { "INPUTS",  "Number of inputs for the CPPN"  },
  { "OUTPUTS", "Number of outputs for the CPPN" },
};

py::dict to_json (const CPPNData &d) {
  py::dict dict;
  py::list nodes, links;
  for (auto &n: d.nodes)  nodes.append(py::make_tuple(n.id, n.func));
  for (auto &l: d.links)
    links.append(py::make_tuple(l.id, l.src, l.dst, l.weight));
  dict["nodes"] = nodes;
  dict["links"] = links;
  dict["nextNodeID"] = d.nextNodeID;
  dict["nextLinkID"] = d.nextLinkID;
  return dict;
}

using Node = CPPNData::Node;
using Link = CPPNData::Link;

CPPNData from_json (py::dict dict) {
  CPPNData d;
  d.nextNodeID = dict["nextNodeID"].cast<int>();
  d.nextLinkID = dict["nextLinkID"].cast<int>();
  for (const py::handle &h: dict["nodes"]) {
    py::tuple t = h.cast<py::tuple>();
    d.nodes.push_back(Node{t[0].cast<int>(), t[1].cast<std::string>()});
  }
  for (const py::handle &h: dict["links"]) {
    py::tuple t = h.cast<py::tuple>();
    d.links.push_back(Link{t[0].cast<int>(),
                            t[1].cast<uint>(), t[2].cast<uint>(),
                            t[3].cast<float>()});
  }
  return d;
}

void init_genotype (py::module_ &m) {
  auto cppn = py::class_<CPPNData>(m, "CPPNData");
  auto node = py::class_<Node>(cppn, "Node");
  auto link = py::class_<Link>(cppn, "Link");

  py::bind_vector<std::vector<Node>>(cppn, "Nodes", "Collection of Nodes");
  py::bind_vector<std::vector<Link>>(cppn, "Links", "Collection of Links");

#define ID(X, ...) (#X, &CLASS::X, ##__VA_ARGS__)
#define CLASS CPPNData
  cppn.doc() = R"(C++ supporting type for genomic data)";
  cppn.def(py::init<>())
      .def_readonly_static ID(INPUTS, "")
      .def_readonly_static ID(OUTPUTS)
      .def_readonly_static("_docstrings", &_cppn_doc)
      .def_readwrite ID(nodes, "The collection of computing nodes")
      .def_readwrite ID(links, "The collection of inter-node relationships")
      .def_readwrite ID(nextNodeID, "ID for the next random node (monotonic)")
      .def_readwrite ID(nextLinkID, "ID for the next random link (monotonic")
      .def("to_json", to_json, "Convert to a json-compliant Python dictionary")
      .def_static("from_json", from_json, "j"_a,
                  "Convert from the json-compliant Python dictionary `j`")
      .def(py::pickle(
        [] (const CLASS &d) { return to_json(d); },
        [](py::dict d) {      return from_json(d);  }
      ))
      ;

#undef CLASS
#define CLASS Node
  node.doc() = "Computational node of a CPPN";
  node.def(py::init<int, const Node::FuncID &>())
      .def("__repr__", [] (const Node &n) {
        std::ostringstream oss;
        oss << "N" << n.id << ":" << n.func;
        return oss.str();
      })
      .def_readwrite ID(id, "Numerical identifier")
      .def_readwrite ID(func, "Function used to compute");

#undef CLASS
#define CLASS Link
  link.doc() = "From-to relationship between two computational node";
  link.def(py::init<int, uint, uint, float>())
      .def("__repr__", [] (const Link &l) {
        std::ostringstream oss;
        oss << "L" << l.id << ":" << l.src << " -(" << l.weight << ")-> "
            << l.dst;
        return oss.str();
      })
      .def_readwrite ID(id, "Numerical identifier")
      .def_readwrite ID(src, "ID of the source node")
      .def_readwrite ID(dst, "ID of the destination node")
      .def_readwrite ID(weight, "Connection weight");
}


} // end of namespace kgd::eshn::pybind
