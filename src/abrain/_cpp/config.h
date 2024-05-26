#ifndef KGD_GENOTYPE_CONFIG_H
#define KGD_GENOTYPE_CONFIG_H

#include <vector>
#include <map>

#include "genotype.h"

namespace kgd::eshn {

template <typename T>
struct Bounds {
  T min, rndMin, rndMax, max, stddev;
};

struct Config {
  /// ================================================
  /// CPPN genotype parameters

  using FID = genotype::CPPNData::Node::FuncID;
  using Functions = std::vector<FID>;

  using FunctionSet = Functions;
  static FunctionSet functionSet;

  static FID defaultOutputFunction;

  enum ESHNOutputs {
    WEIGHT, LEO, BIAS
  };

  using OutputFunctions = std::map<ESHNOutputs, FID>;
  static OutputFunctions eshnOutputFunctions;

  using MutationRates = std::map<std::string, float>;
  static MutationRates mutationRates;

  using FBounds = Bounds<float>;
  static FBounds cppnWeightBounds;

  /// ================================================
  /// ANN parameters
  static float annWeightsRange;
  static FID activationFunc;

  /// ================================================
  /// ES-HyperNEAT parameters
  static unsigned int initialDepth, maxDepth, iterations;
  static float divThr, varThr, bndThr;
  static bool allowPerceptrons;
};

} // end of namespace kgd::eshn

#endif // KGD_GENOTYPE_CONFIG_H
