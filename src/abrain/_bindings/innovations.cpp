#include "../_cpp/innovations.h"
#include "utils.hpp"

#include "pybind11/pybind11.h"
namespace py = pybind11;
using namespace pybind11::literals;

#include "pybind11/stl_bind.h"

using namespace kgd::eshn::genotype;
namespace kgd::eshn::pybind {

using ID = Innovations::ID;
using Key = Innovations::Key;
using Map = Innovations::Map;

py::dict to_json (const Innovations &i) {
  py::dict dict;
  static const auto to_json = [] (const Map &m) {
      py::dict dict;
      for (const auto &[key, value]: m) {
        dict[py::make_tuple(key.first, key.second)] = value;
      }
      return dict;
  };
  dict["nodes"] = to_json(i.nodes());
  dict["links"] = to_json(i.links());
  dict["nextNode"] = i.nextNodeID();
  dict["nextLink"] = i.nextLinkID();
  return dict;
}

auto innovations_from_json (const py::dict& dict) {
  static const auto from_json = [] (const py::dict &dict) {
    Map m;
    for (const auto &[p_key, p_value]: dict) {
      const auto t = p_key.cast<py::tuple>();
      const auto key = std::make_pair(t[0].cast<ID>(), t[1].cast<ID>());
      m[key] = p_value.cast<ID>();
    }
    return m;
  };
  auto nodes = from_json(dict["nodes"]),
       links = from_json(dict["links"]);
  auto nextNodeID = dict["nextNode"].cast<ID>(),
       nextLinkID = dict["nextLink"].cast<ID>();
  return Innovations(std::move(nodes), std::move(links), nextNodeID, nextLinkID);
}

void init_innovations (py::module_ &m) {
  auto inov = py::class_<Innovations>(m, "Innovations");
//  auto node = py::bind_map<Map>(inov, "Map");

#define ID(X, ...) (#X, &CLASS::X, __VA_ARGS__)
#define CLASS Innovations
  inov.doc() = R"(C++ database for innovation markings)";
  inov.def(py::init<>())

          .def ID(initialize,
                  R"__(
            Reset the database.

            Next node id is set to the provided value and the next link
            id is set to 0. All known nodes and links mappings are cleared.

            :param nextNodeID: The next node id to give
                  )__", "nextNodeID"_a)

          .def ID(link_id, R"__(
            Retrieve link innovation marking for provided key

            :param src: ID of the link's source
            :param dst: ID of the link's destination
            :return: The corresponding *link* innovation marking or
              :attr:`Innovations.NOT_FOUND` if not found
                  )__", "src"_a, "dst"_a)
          .def ID(node_id, R"__(
            Retrieve node innovation marking for provided key

            The link refers to the link that was broken up to create this new
             node

            :param src: ID of the link's source
            :param dst: ID of the link's destination
            :return: The corresponding *node* innovation marking or
              :attr:`Innovations.NOT_FOUND` if not found
                  )__", "src"_a, "dst"_a)

          .def ID(get_link_id, R"__(
            Attempt to generate a new link innovation marking for provided key

            :param src: ID of the link's source
            :param dst: ID of the link's destination
            :return: The newly created, or existing corresponding *link*
              innovation marking
                  )__", "src"_a, "dst"_a)
          .def ID(get_node_id, R"__(
            Attempt to generate a new node innovation marking for provided key

            :param src: ID of the link's source
            :param dst: ID of the link's destination
            :return: The newly created, or existing corresponding *node*
              innovation marking
                  )__", "src"_a, "dst"_a)

          .def ID(new_link_id, R"__(
            Force generation of a new link innovation marking for provided key

            :param src: ID of the link's source
            :param dst: ID of the link's destination
            :return: The newly created *link* innovation marking
                  )__", "src"_a, "dst"_a)
          .def ID(new_node_id, R"__(
            Force generation of a new node innovation marking for provided key

            :param src: ID of the link's source
            :param dst: ID of the link's destination
            :return: The newly created *node* innovation marking
                  )__", "src"_a, "dst"_a)

          .def("next_node_id", &Innovations::nextNodeID,
               "Historical marking of the next new node")
          .def ("next_link_id", &Innovations::nextLinkID,
                "Historical marking of the next new link")

          .def_readonly_static
            ID(NOT_FOUND, "Returned when requested id could not be found")

          .def("__repr__", [] (const Innovations &i) {
            return utils::mergeToString(
                    "Innovations(", i.nodes().size(), ", ", i.links().size(),
                    ")");
          })

          .def("empty", [] (const Innovations &i) {
            return i.nodes().empty() && i.links().empty();
          }, "Whether any historical markings have been generated yet")
          .def("size", [] (const Innovations &i) {
            return py::make_tuple(i.nodes().size(), i.links().size());
          }, "The number of node and link historical markings currently"
             " registered")

          .def ID(copy,
                  "Return a perfect (deep)copy of this innovations database")

          .def("to_json", to_json, "Convert to a json-compliant Python"
                                   " dictionary")
          .def_static("from_json", innovations_from_json, "j"_a,
                      "Convert from the json-compliant Python dictionary `j`")
          .def(py::pickle(
                  [](const CLASS &d) { return to_json(d); },
                  [](const py::dict &d) { return innovations_from_json(d);  }
          ))
          ;
}


} // end of namespace kgd::eshn::pybind
