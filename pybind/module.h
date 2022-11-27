#ifndef PyPP_INTERFACE_H
#define PyPP_INTERFACE_H

#include "pybind11/pybind11.h"

/* Need to expose:
 * - develop(CPPN_raw) -> ANN
 */

namespace kgd::eshn {

void develop (void);

} // end of namespace kgd::eshn

#endif // PyPP_INTERFACE_H
