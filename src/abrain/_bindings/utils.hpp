#ifndef KGD_ESHN_BINDING_UTILS_HPP
#define KGD_ESHN_BINDING_UTILS_HPP

#include <sstream>
#include <map>

#include "pybind11/stl.h"

#include "../_cpp/misc/utils.hpp"

namespace kgd::eshn::utils {

namespace py = pybind11;

using DocMap = std::map<std::string, std::string>;

template <typename B>
std::string toString (const char *name, const B &b) {
  std::ostringstream oss;
  oss << name << "[";
  for (const auto &v: b) oss << " " << v;
  oss << " ]";
  return oss.str();
}

#ifndef NDEBUG
inline py::tuple tuple(const std::vector<float> &v) {
  return {py::cast(v)};
}

inline void set_to_nan(std::vector<float> &v) {
  std::ranges::fill(v.begin(), v.end(), NAN);
}

inline bool valid(const std::vector<float> &v) {
  return std::ranges::none_of(v.begin(), v.end(),
                              static_cast<bool(*)(float)>(std::isnan));
}

static constexpr auto doc_tuple = "Debug helper to convert to Python tuple";
static constexpr auto doc_set_to_nan = "Debug helper to set all values to NaN";
static constexpr auto doc_valid = "Debug tester to assert no values are NaN";
#endif

template <typename B>
void init_buffer (py::handle scope, const char *name, const char *doc) {
  py::options options;
  options.disable_function_signatures();

  auto buff = py::class_<B, std::shared_ptr<B>>(scope, name);
  buff.doc() = doc;
  buff.def("__len__", [] (const B &b) { return b.size(); })
      .def("__repr__", [name] (const B &b) { return toString(name, b); })
      .def("__iter__", [] (const B &b) {
        return py::make_iterator(b.begin(), b.end());
      })
      .def("__setitem__",
           [] (B &b, size_t i, float v) { b[i] = v; },
           "Assign an element")
      .def("__setitem__",
           [](B &b, const py::slice &slice, const py::iterable &items) {
               size_t start = 0, stop = 0, step = 0, slicelength = 0;
               if (!slice.compute(b.size(), &start, &stop, &step, &slicelength))
                 throw py::error_already_set();
               if (slicelength != py::len(items))
                 throw std::runtime_error("Left and right hand size of slice"
                                          " assignment have different sizes!");
               for (py::handle h: items) {
//           for (size_t i = 0; i < slicelength; ++i) {
                 b[start] = h.cast<float>();
                 start += step;
               }
      })
      .def("__getitem__",
           [] (const B &b, size_t i) { return b[i]; },
           "Access an element")
      .def("__getitem__",
           [](const B &buf, const py::slice &slice) -> py::list * {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(buf.size(), &start, &stop, &step, &slicelength))
              throw py::error_already_set();

            auto *list = new py::list(slicelength);
            for (size_t i = 0; i < slicelength; ++i) {
              (*list)[i] = buf[start];
              start += step;
            }
            return list;
      })
#ifndef NDEBUG
      .def("tuple", [] (const B &b) { return tuple(b); }, doc_tuple)
      .def("set_to_nan", [] (B &b) { return set_to_nan(b); }, doc_set_to_nan)
      .def("valid", [] (const B &b) { return valid(b); }, doc_valid)
#endif
      ;
}

} // end of namespace kgd::eshn::utils

#endif // KGD_ESHN_BINDING_UTILS_HPP
