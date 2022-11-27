#ifndef KGD_GENOTYPE_CONFIG_H
#define KGD_GENOTYPE_CONFIG_H

#include <vector>
#include <set>
#include <map>

#include "genotype.h"

namespace kgd::eshn {

template <typename T>
struct Bounds {
  T min, rndMin, rndMax, max;
};

struct Config {
  using FID = genotype::CPPNData::Node::FuncID;
  using Functions = std::vector<FID>;

  using FunctionSet = Functions;
  static FunctionSet functionSet;

  using OutputFunctions = Functions;
  static OutputFunctions outputFunctions;

  using MutationRates = std::map<std::string, float>;
  static MutationRates mutationRates;

  using FBounds = Bounds<float>;
  static FBounds weightBounds;
};

} // end of namespace kgd::eshn

#endif // KGD_GENOTYPE_CONFIG_H
