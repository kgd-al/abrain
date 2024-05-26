#ifndef KGD_CPPN_GENOTYPE_H
#define KGD_CPPN_GENOTYPE_H

#include <string>
#include <vector>
#include <map>

namespace kgd::eshn::genotype {

struct CPPNData {
  using ID = int;

  struct Node {
    ID id;

    using FuncID = std::string;
    FuncID func;
  };

  struct Link {
    ID id;

    unsigned int src, dst;
    float weight;
  };

  unsigned int inputs;
  unsigned int outputs;
  std::string labels;
  bool bias;

  std::vector<Node> nodes;
  std::vector<Link> links;
};

} // end of namespace kgd::eshn::genotype

#endif // KGD_CPPN_GENOTYPE_H
