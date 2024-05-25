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

py::dict to_json (const CPPNData &d) {
  py::dict dict;
  py::list nodes, links;
  for (const auto &[id, func]: d.nodes)  nodes.append(py::make_tuple(id, func));
  for (const auto &[id, src, dst, weight]: d.links)
    links.append(py::make_tuple(id, src, dst, weight));
  dict["inputs"] = d.inputs;
  dict["outputs"] = d.outputs;
  dict["bias"] = d.bias;
  dict["labels"] = d.labels;
  dict["nodes"] = nodes;
  dict["links"] = links;
  return dict;
}

using Node = CPPNData::Node;
using Link = CPPNData::Link;

CPPNData genotype_from_json (const py::dict& dict) {
  CPPNData d;
  d.inputs = dict["inputs"].cast<int>();
  d.outputs = dict["outputs"].cast<int>();
  d.bias = dict["bias"].cast<bool>();
  d.labels = dict["labels"].cast<std::string>();
  for (const py::handle &h: dict["nodes"]) {
    auto t = h.cast<py::tuple>();
    d.nodes.push_back(Node{t[0].cast<int>(), t[1].cast<std::string>()});
  }
  for (const py::handle &h: dict["links"]) {
    auto t = h.cast<py::tuple>();
    d.links.push_back(Link{t[0].cast<int>(),
                            t[1].cast<unsigned int>(), t[2].cast<unsigned int>(),
                            t[3].cast<float>()});
  }
  return d;
}

template <typename T>
void id_sorted(std::vector<T> &v) {
  std::sort(v.begin(), v.end(), [] (const T &lhs, const T &rhs) {
    return lhs.id < rhs.id;
  });
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
      .def_readwrite ID(inputs, "Number of inputs")
      .def_readwrite ID(outputs, "Number of outputs")
      .def_readwrite ID(bias, "Whether to use an input bias")
      .def_readwrite ID(labels, "(optional) label for the inputs/outputs")
      .def_readwrite ID(nodes, "The collection of computing nodes")
      .def_readwrite ID(links, "The collection of inter-node relationships")

      .def(
        "_sort_by_id",
        [] (CPPNData &d) {
          id_sorted(d.nodes);
          id_sorted(d.links);
        }, "Ensures both nodes and links are id-sorted")

      .def("to_json", to_json, "Convert to a json-compliant Python dictionary")
      .def_static("from_json", genotype_from_json, "j"_a,
                  "Convert from the json-compliant Python dictionary `j`")
      .def(py::pickle(
        [](const CLASS &d) { return to_json(d); },
        [](const py::dict &d) { return genotype_from_json(d);  }
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
      .def_readwrite ID(id, "Historical marking")
      .def_readwrite ID(func, "Function used to compute");

#undef CLASS
#define CLASS Link
  link.doc() = "From-to relationship between two computational node";
  link.def(py::init<int, unsigned int, unsigned int, float>())
      .def("__repr__", [] (const Link &l) {
        std::ostringstream oss;
        oss << "L" << l.id << ":" << l.src << " -(" << l.weight << ")-> "
            << l.dst;
        return oss.str();
      })
      .def_readwrite ID(id, "Historical marking")
      .def_readwrite ID(src, "ID of the source node")
      .def_readwrite ID(dst, "ID of the destination node")
      .def_readwrite ID(weight, "Connection weight");
}


} // end of namespace kgd::eshn::pybind
