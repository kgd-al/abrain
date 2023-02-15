#include "config.h"

namespace kgd::eshn {

/// ================================================
/// CPPN genotype parameters

Config::FunctionSet Config::functionSet {
  "abs", "gaus", "id", "bsgm", "sin", "step"
};

Config::OutputFunctions Config::outputFunctions {
  "bsgm", "step", "id"
};

template<typename K, typename V>
std::map<K,V> normalize (std::initializer_list<std::pair<K, V>> l) {
  V sum = 0;
  std::map<K,V> map;
  for (const auto &p: l)  sum += p.second;
  for (auto &p: l) map[p.first] = p.second / sum;
  return map;
} // LCOV_EXCL_LINE

Config::MutationRates Config::mutationRates =
    normalize<std::string, float>({
                { "add_n",  .5f   },
                { "add_l",  .75f  },
                { "del_n",  .75f  },
                { "del_l", 1.0f   },
                { "mut_w", 5.5f   },
                { "mut_f", 2.5f   },
});

Config::FBounds Config::cppnWeightBounds {-3.f, -1.f, 1.f, 3.f, 0.01f};

/// ================================================
/// ANN parameters

float Config::annWeightsRange = 3;
Config::FID Config::activationFunc = "ssgn";

/// ================================================
/// ES-HyperNEAT parameters

uint Config::initialDepth = 2;
uint Config::maxDepth = 3;
uint Config::iterations = 10;

float Config::divThr = .3f;
float Config::varThr = .3f;
float Config::bndThr = .15f;

bool Config::allowPerceptrons = true;

} // end of namespace kgd::eshn
