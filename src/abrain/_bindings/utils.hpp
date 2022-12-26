#ifndef KGD_ESHN_BINDING_UTILS_HPP
#define KGD_ESHN_BINDING_UTILS_HPP

#include <ostream>
#include <utility>
#include <map>

#include "pybind11/stl.h"

namespace kgd::eshn::utils {

/// Builds a string from the arguments, delegating type conversion to
///  appropriate operator<<
template <typename... ARGS>
std::string mergeToString (ARGS... args) {
  std::ostringstream oss;
  (oss << ... << std::forward<ARGS>(args));
  return oss.str();
}

using DocMap = std::map<std::string, std::string>;

} // end of namespace kgd::eshn::utils

#endif // KGD_ESHN_BINDING_UTILS_HPP
