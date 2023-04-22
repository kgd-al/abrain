#include "../_cpp/config.h"
#include "../_cpp/phenotype/cppn.h"
#include "utils.hpp"

#include <iostream>

#include "pybind11/pybind11.h"
namespace py = pybind11;
using namespace pybind11::literals;

#include "pybind11/stl_bind.h"
using namespace kgd::eshn;

PYBIND11_MAKE_OPAQUE(Config::Functions)
PYBIND11_MAKE_OPAQUE(Config::MutationRates)

namespace kgd::eshn::pybind {

using Sections = std::map<std::string, std::vector<std::string>>;
static const Sections _sections {
  { "CPPN", {
    "functionSet",
    "outputFunctions",
    "mutationRates",
    "cppnWeightBounds",
  }},

  { "ANN", {
    "annWeightsRange",
    "activationFunc",
  }},

  { "ESHN", {
    "initialDepth",
    "maxDepth",
    "iterations",
    "divThr",
    "varThr",
    "bndThr",
    "allowPerceptrons"
  }}
};

static utils::DocMap _docs {
  { "functionSet",
    "List of functions accessible to nodes via creation/mutation" },

  { "cppnWeightBounds",
    "Initial and maximal bounds for each of the CPPN's weights" },

  { "mutationRates",
    "Probabilities for each point mutation (addition/deletion/alteration)\n\n"
    "Glossary:\n"
    "  - add_l: add a random link between two nodes (feedforward only)\n"
    "  - add_n: replace a link by creating a node\n"
    "  - del_l: delete a random link (never leaves unconnected nodes)\n"
    "  - del_n: replace a simple node by a direct link\n"
    "  - mut_f: change the function of a node\n"
    "  - mut_w: change the connection weight of a link\n\n"
  },

  { "cppnInputNames",
    "``const`` Auto generated name of the CPPN inputs (based on dimensions and"
    " optional use of the connection length)"},

  { "cppnOutputNames",
    "``const`` Auto generated name of the CPPN outputs" },

  { "outputFunctions",
    "Functions used for the CPPN output (same length as"
    " :attr:`cppnOutputNames`)" },

  { "activationFunc",
    "The activation function used by all hidden/output neurons"
    " (inputs are passthrough)" },

  { "annWeightsRange",
    "Scaling factor `s` for the CPPN `w` output mapping"
    " :math:`[-1,1] \to [-s,s]`" },

  { "initialDepth",
    "Initial division depth for the underlying quad-/octtree" },

  { "maxDepth",
    "Maximal division depth for the underlying quad-/octtree" },

  { "iterations",
    "Maximal number of discovery steps for Hidden/Hidden connections."
    " Can stop early in case of convergence (no new neurons discovered)" },

  { "divThr",
    "Division threshold for a quad-/octtree cell/cube" },

  { "varThr",
    "Variance threshold for exploring a quad-/octtree cell/cube" },

  { "bndThr",
    "Minimal divergence threshold for discovering neurons" },

  { "allowPerceptrons",
    "Attempt to generate a perceptron if no hidden neurons were discovered" },
};

void init_config (py::module_ &m) {
  using Strings = std::vector<std::string>;

  auto strs = py::bind_vector<Strings>(m, "Strings");
  strs.doc() = "C++ list of strings";

  if (!std::is_same<Strings, Config::Functions>::value)
    py::bind_vector<Config::Functions>(m, "Functions");

  auto mutr = py::bind_map<Config::MutationRates>(m, "MutationRates");
  mutr.doc() = "C++ mapping between mutation types and rates";

  auto fbnd = py::class_<Config::FBounds>(m, "FBounds");
  fbnd.doc() = "C++ encapsulation for mutation bounds";

  auto cnfg = py::class_<Config>(m, "Config");

#define ID(X, ...) (#X, &CLASS::X, ##__VA_ARGS__)
#define CLASS Config
  cnfg.def_property_readonly_static("cppnInputNames", [] (py::object){
        Strings names;
        for (uint i=0; i<cppn::CPPN_INPUTS; i++)
          names.push_back(std::string(cppn::CPPN_INPUT_NAMES[i]));
        return names;
      })
      .def_property_readonly_static("cppnOutputNames", [] (py::object){
        Strings names;
        for (uint i=0; i<cppn::CPPN_OUTPUTS; i++)
          names.push_back(std::string(cppn::CPPN_OUTPUT_NAMES[i]));
        return names;
      })
      .def_readwrite_static ID(outputFunctions)

      .def_readwrite_static ID(functionSet)
      .def_readwrite_static ID(cppnWeightBounds)
      .def_readwrite_static ID(mutationRates)

      .def_readwrite_static ID(annWeightsRange)
      .def_readwrite_static ID(activationFunc)

      .def_readwrite_static ID(initialDepth)
      .def_readwrite_static ID(maxDepth)
      .def_readwrite_static ID(iterations)
      .def_readwrite_static ID(divThr)
      .def_readwrite_static ID(varThr)
      .def_readwrite_static ID(bndThr)
//       .def_readwrite_static ID(allowPerceptrons)
      .def_property_static( // More verbose but prevents accidental conversions
        "allowPerceptrons",
        [](py::object) { return Config::allowPerceptrons; },
        [](py::object, bool b) { Config::allowPerceptrons = b; })

      .def_readonly_static("_sections", &_sections)
      .def_readonly_static("_docstrings", &_docs)

      .def_static("known_function", [] (const phenotype::CPPN::FuncID &f) {
        static const auto &funcs = phenotype::CPPN::functions;
        return funcs.find(f) != funcs.end();
      }, "Whether the requested function name is a built-in", "name"_a)
  ;

  cnfg.doc() = R"(C++/Python configuration values for the ABrain library)";


#undef CLASS
#define CLASS Config::FBounds
  fbnd.def(py::init<>())
      .def(py::init<float, float, float, float, float>())
      .def_readwrite ID(rndMin).def_readwrite ID(min)
      .def_readwrite ID(rndMax).def_readwrite ID(max)
      .def_readwrite ID(stddev)
      .def("__eq__",
           [] (const Config::FBounds &lhs, const Config::FBounds &rhs) {
              return lhs.min == rhs.min
                  && lhs.rndMin == rhs.rndMin
                  && lhs.rndMax == rhs.rndMax
                  && lhs.max == rhs.max
                  && lhs.stddev == rhs.stddev;
      })
      .def("__repr__", [] (const Config::FBounds &fb) {
        std::ostringstream oss;
        oss << "Bounds(" << fb.min << ", " << fb.rndMin << ", " << fb.rndMax
            << ", " << fb.max << ", " << fb.stddev << ")";
        return oss.str();
      })
  ;

  // json converter (read/write to primitive python types)
  fbnd.def("toJson", [] (const Config::FBounds &b) {
      std::vector<float> v {{b.min, b.rndMin, b.rndMax, b.max, b.stddev}};
        return py::list(py::cast(v));
  }, "Convert to a python list of floats")
      .def_static("fromJson", [] (const std::vector<float> &l) {
        return Config::FBounds{l[0], l[1], l[2], l[3], l[4]};
  }, "Convert from a python list of floats")
      .def("isValid", [] (const Config::FBounds &b) {
    return b.min <= b.rndMin && b.rndMin <= b.rndMax && b.rndMax <= b.max
        && b.stddev > 0;
  }, "Whether this is a valid mutation bounds object");

  strs.def("toJson", [] (const Strings &s) {
      return py::list(py::cast(s));
  }, "Convert to a python list of strings")
      .def_static("fromJson", [] (const py::list &l) {
        Strings s;
        for (auto item: l)  s.push_back(item.cast<std::string>());
        return s;
  }, "Convert from a python list of strings")
      .def("isValid", [] (const Strings &s) {
    return !s.empty();
  }, "Whether this is a valid strings colleciton (not empty)");


  mutr.def("toJson", [] (const Config::MutationRates &s) {
      return py::dict(py::cast(s));
  }, "Convert to a python map of strings/float")
      .def_static("fromJson", [] (const py::dict &d) {
        Config::MutationRates r;
        for (auto pair: d)
          r[pair.first.cast<std::string>()] = pair.second.cast<float>();
        return r;
  }, "Convert from a python map of strings/floats")
      .def("isValid", [] (const Config::MutationRates &r) {
    float sum = 0;
    for (const auto &p: r) {
      if (p.second < 0) return false;
      sum += p.second;
    }
    return sum > 0;
  }, "Whether this is a valid dictionary of mutation rates");
}

} // end of namespace kgd::eshn::pybind
