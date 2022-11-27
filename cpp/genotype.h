#ifndef KGD_CPPN_GENOTYPE_H
#define KGD_CPPN_GENOTYPE_H

#include <string>
#include <vector>

#include "misc/constants.h"

namespace kgd::eshn::genotype {

template <uint I, uint O>
struct CPPNData_Template {
  using Input = kgd::eshn::cppn::CPPN_INPUT;
  using Output = kgd::eshn::cppn::CPPN_OUTPUT;

  struct Node {
    using FuncID = std::string;
    FuncID func;
  };

  struct Link {
    uint src, dst;
    float weight;
  };

  static constexpr uint INPUTS = I;
  static constexpr uint OUTPUTS = O;
  std::vector<Node> nodes;
  std::vector<Link> links;
};
using CPPNData = CPPNData_Template<kgd::eshn::cppn::CPPN_INPUTS,
                                   kgd::eshn::cppn::CPPN_OUTPUTS>;

} // end of namespace kgd::eshn::genotype

#endif // KGD_CPPN_GENOTYPE_H
