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
PYBIND11_MAKE_OPAQUE(Config::OutputFunctions)

namespace kgd::eshn::pybind {

template <typename T>
T pyStringToEnum(const py::enum_<T>& enm, const std::string& value) {
  auto values = enm.attr("__members__").template cast<py::dict>();
  auto strVal = py::str(value);
  if (values.contains(strVal))
    return T(values[strVal].template cast<T>());
  throw "Invalid string value " + value + " for enum " + std::string(typeid(T).name());
}

template <typename T>
py::str enumToPyString(const py::enum_<T>& enm, const T& value) {
  auto values = enm.attr("__members__").template cast<py::dict>();
  for (auto val : values)
    if (T(val.second.template cast<T>()) == value)
      return py::str(val.first);
  throw "Invalid value for enum " + std::string(typeid(T).name());
}

template <typename M>
M fromDict(const py::dict &d) {
  M map;
  for (const auto &pair: d)
    map[pair.first.cast<typename M::key_type>()]
      = pair.second.cast<typename M::mapped_type>();
  return map;
}

template <typename T>
std::ostream& operator<<(std::ostream &os, const std::vector<T> &vector) {
  os << "[";
  if (!vector.empty())
    for (const auto &v: vector) os << " " << v;
  return os << " ]";
}

template <typename T>
bool vectorFind(const std::vector<T> &vector, const T &value) {
  for (const auto &v: vector) if (v == value) return true;
  return false;
}

py::object get_pybind11_metaclass() {
  auto &internals = py::detail::get_internals();
  return py::reinterpret_borrow<py::object>((PyObject*)internals.default_metaclass);
}
py::object get_standard_metaclass() {
  return py::reinterpret_borrow<py::object>((PyObject *)&PyType_Type);
}

py::object create_enum_metaclass() {
  py::dict attributes;
  attributes["__len__"] = py::cpp_function(
          [](py::object cls) {
              return py::len(cls.attr("__entries"));
          }
          , py::is_method(py::none())
  );
  attributes["__iter__"] = py::cpp_function(
          [](py::object cls) {
              auto list = new py::list();
              for (const auto &pair: cls.attr("__entries").cast<py::dict>())
                list->append(pair.second.cast<py::tuple>()[0]);
              return py::make_iterator<py::return_value_policy::take_ownership>(*list);
          }
          , py::is_method(py::none())
  );
  auto pybind11_metaclass = get_pybind11_metaclass();
  auto standard_metaclass = get_standard_metaclass();
  return standard_metaclass(std::string("pybind11_ext_enum"),
                            py::make_tuple(pybind11_metaclass), attributes);
}

using Sections = std::map<std::string, std::vector<std::string>>;
static const Sections _sections {
  { "CPPN", {
    "functionSet",
    "defaultOutputFunction",
    "eshnOutputFunctions",
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

  { "defaultOutputFunction",
    "Output function for random generic CPPNs" },

  { "eshnOutputFunctions",
    "Functions used for :class:`CPPN2D` and :class:`CPPN3D` outputs" },

  { "activationFunc",
    "The activation function used by all hidden/output neurons"
    " (inputs are passthrough)" },

  { "annWeightsRange",
    "Scaling factor `s` for the CPPN `w` output mapping"
    " :math:`[-1,1] to [-s,s]`" },

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
  auto cnfg = py::class_<Config>(m, "Config");

  py::options options;
  options.disable_function_signatures();

  using Strings = std::vector<std::string>;
  auto strs = py::bind_vector<Strings>(cnfg, "Strings");
  strs.doc() = "C++ list of strings";

  if constexpr (!std::is_same_v<Strings, Config::Functions>)
    py::bind_vector<Config::Functions>(cnfg, "Functions");

  auto mutr = py::bind_map<Config::MutationRates>(m, "MutationRates");
  mutr.doc() = "C++ mapping between mutation types and rates";

  auto fbnd = py::class_<Config::FBounds>(cnfg, "FBounds");
  fbnd.doc() = "C++ encapsulation for mutation bounds";

  auto enum_metaclass = create_enum_metaclass();
  auto eshn = py::enum_<Config::ESHNOutputs>(cnfg, "ESHNOutputs", py::metaclass(enum_metaclass));

  auto fout = py::bind_map<Config::OutputFunctions>(m, "OutputFunctions");
  fout.doc() = "C++ mapping between CPPN's outputs, when used with ES-HyperNEAT, and functions";

  options.enable_function_signatures();

#define ID(X, ...) (#X, &CLASS::X, ##__VA_ARGS__)
#define CLASS Config
  cnfg.def_readwrite_static ID(eshnOutputFunctions)
      .def_readwrite_static ID(defaultOutputFunction)

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
        [](const py::object&) { return Config::allowPerceptrons; },
        [](const py::object&, bool b) { Config::allowPerceptrons = b; })

      .def_readonly_static("_sections", &_sections)
      .def_readonly_static("_docstrings", &_docs)

      .def_static("known_function", [] (const phenotype::CPPN::FuncID &f) {
        static const auto &funcs = phenotype::CPPN::functions;
        return funcs.find(f) != funcs.end();
      }, "Whether the requested function name is a built-in", "name"_a)

      .def_static("test_valid", [] () {
        bool show_all_functions = false, show_current_functions = false;
        int errors = 0;
        std::ostringstream oss;

        const auto &all_functions = phenotype::CPPN::functions;
        if (all_functions.find(Config::activationFunc) == all_functions.end()) {
          oss << "> " << "requested activation function '" << Config::activationFunc
              << "' is not implemented. Error or potential PR?\n";
          show_all_functions = true;
          errors++;
        }

        for (const auto &f: Config::functionSet) {
          if (all_functions.find(f) == all_functions.end()) {
            oss << "> " << "requested function '" << f
                << "' is not implemented. Error or potential PR?\n";
            show_all_functions = true;
            errors++;
          }
        }

        if (all_functions.find(Config::activationFunc) == all_functions.end()) {
          oss << "> " << "requested activation function '" << Config::activationFunc
              << "' is not implemented. Error or potential PR?\n";
          show_all_functions = true;
          errors++;
        }

        if (!vectorFind(Config::functionSet, Config::defaultOutputFunction)) {
          oss << "> " << "Default output function '" << Config::defaultOutputFunction
              << "' not found in current function set\n";
          show_current_functions = true;
          errors++;
        }

        for (const auto &eof_item: Config::eshnOutputFunctions) {
          if (!vectorFind(Config::functionSet, eof_item.second)) {
            oss << "> " << "ES-HyperNEAT output function '" << eof_item.second
                << "' not found in current function set\n";
            show_current_functions = true;
            errors++;
          }
        }

        if (errors > 0) {
          std::ostringstream _oss;
          _oss << "Found " << errors << " errors when testing current configuration:\n";
          _oss << oss.str();
          if (show_all_functions) {
            std::vector<std::string> af;
            for (const auto &p: all_functions) af.push_back(p.first);
            _oss << "All implemented functions: " << af << "\n";
          }
          if (show_current_functions)
            _oss << "Currently selected functions: " << Config::functionSet << "\n";
          throw std::domain_error(_oss.str());
        }

        return errors == 0;
      }, "Function used after reading a configuration file to ensure validity");
  ;

  cnfg.doc() = R"(C++/Python configuration values for the ABrain library)";

  eshn.value("Weight", Config::ESHNOutputs::WEIGHT)
      .value("LEO", Config::ESHNOutputs::LEO)
      .value("Bias", Config::ESHNOutputs::BIAS)
      .export_values()
      .def_static("__iter__", [eshn] () {
        return py::dict(eshn.attr("__entries"));
      })
      ;
  eshn.attr("name").doc() = "Name of the output";
  eshn.attr("value").doc() = "Corresponding value of the output";
  py::implicitly_convertible<std::string, Config::ESHNOutputs>();

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
      }, "Whether this is a valid strings collection (not empty)");

  mutr.def(py::init(&fromDict<Config::MutationRates>))
      .def("toJson", [] (const Config::MutationRates &s) {
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

  fout.def(py::init(&fromDict<Config::OutputFunctions>))
      .def("toJson", [eshn] (const Config::OutputFunctions &f) {
        auto dict = py::dict();
        for (const auto& pair: f)
          dict[enumToPyString(eshn, pair.first)] = py::cast(pair.second);
        return dict;
      }, "Convert to a python dict")
      .def_static("fromJson", [eshn] (const py::dict &d) {
        Config::OutputFunctions f;
        for (auto pair: d)
          f[pyStringToEnum(eshn, pair.first.cast<std::string>())] = pair.second.cast<std::string>();
        return f;
      }, "Convert from a python dict")
      .def("isValid", [eshn] (const Config::OutputFunctions &f) {
        if (f.size() != py::len(eshn.attr("__entries"))) {
          std::cerr << "Missing entries" << std::endl;
          return false;
        }
        return true;
      }, "Whether this is a collection of output functions");
      ;
}

} // end of namespace kgd::eshn::pybind
