#ifndef KGD_CPPN_GENOTYPE_H
#define KGD_CPPN_GENOTYPE_H

#include <string>
#include <vector>

namespace kgd::eshn::genotype {

struct CPPNData {
  struct Node {
    int id;

    using FuncID = std::string;
    FuncID func;
  };

  struct Link {
    int id;

    uint src, dst;
    float weight;
  };

  uint inputs;
  uint outputs;
  std::string labels;

  std::vector<Node> nodes;
  std::vector<Link> links;

  int nextNodeID, nextLinkID;
};

} // end of namespace kgd::eshn::genotype

#endif // KGD_CPPN_GENOTYPE_H
