#ifndef KGD_ESHN_CPP_UTILS_HPP
#define KGD_ESHN_CPP_UTILS_HPP

#include <sstream>

namespace kgd::eshn::utils {

/// Builds a string from the arguments, delegating type conversion to
///  appropriate operator<<
template <typename... ARGS>
std::string mergeToString (ARGS... args) {
    std::ostringstream oss;
    (oss << ... << std::forward<ARGS>(args));
    return oss.str();
}

} // end of namespace kgd::eshn::utils

#endif // KGD_ESHN_CPP_UTILS_HPP
