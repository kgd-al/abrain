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
  }}
};

using ParseInserter = std::function<void(const std::string&)>;
bool do_parse_string (const std::string &s, ParseInserter inserter,
                      const std::string &delims) {
  auto l = s.find_first_of(delims[0]), r = s.find_last_of(delims[1]);
  if (l != std::string::npos && r != std::string::npos) {
    std::stringstream ss (s.substr(l+1, r-l-1));
    std::string item;
    while (std::getline(ss, item, ',')) {
      if (item[0] == ' ') item = item.substr(1);
      inserter(item);
    }

  } else
    return false;
  return true;
}

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
};

void init_config (py::module_ &m) {
  using Strings = std::vector<std::string>;

  auto strs = py::bind_vector<Strings>(m, "Strings");
  if (!std::is_same<Strings, Config::Functions>::value)
    py::bind_vector<Config::Functions>(m, "Functions");

  auto mutr = py::bind_map<Config::MutationRates>(m, "MutationRates");
  auto fbnd = py::class_<Config::FBounds>(m, "FBounds");

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

  /// String parser methods (for config file reading)
  fbnd.def_static("ccpParseString", [] (const std::string &s) {
    Config::FBounds b {};
    std::vector<float> values;
    bool ok = do_parse_string(s, [&values] (auto item) {
      std::istringstream iss (item);
      float f;
      iss >> f;
      values.push_back(f);
    }, "()");

    ok &= (values.size() == 5);
    if (ok) {
      b.min = values[0], b.rndMin = values[1],
      b.rndMax = values[2], b.max = values[3],
      b.stddev = values[4];
    }
    return b;
  }, "Try to parse the provided string into a mutation bounds object")
      .def_static("isValid", [] (const Config::FBounds &b) {
    return b.min <= b.rndMin && b.rndMin <= b.rndMax && b.rndMax <= b.max
        && b.stddev > 0;
  }, "Whether this is a valid mutation bounds object");

  strs.def_static("ccpParseString", [] (const std::string &s) {
    Strings strs;
    bool ok = do_parse_string(s, [&strs] (auto item) {
      strs.push_back(item);
    }, "[]");

    if (ok) return strs;
    else    return Strings{};
  }, "Try to parse the provided string into a strings collection")
      .def_static("isValid", [] (const Strings &s) {
    return !s.empty();
  }, "Whether this is a valid strings colleciton (not empty)");

  mutr.def_static("ccpParseString", [] (const std::string &s) {
    Config::MutationRates mr;
    bool ok = do_parse_string(s, [&mr] (auto item) {
      auto d = item.find_first_of(':');
      auto key = item.substr(0, d), value = item.substr(d+2);

      std::istringstream iss (value);
      float f;
      iss >> f;
      mr[key] = f;
    }, "{}");

    if (ok) return mr;
    else    return Config::MutationRates{};
  }, "Try to parse the provided string into a dictionary of mutation rates")
      .def_static("isValid", [] (const Config::MutationRates &r) {
    return !r.empty();
  }, "Whether this is a valid dictionary of mutation rates");
}

} // end of namespace kgd::eshn::pybind
