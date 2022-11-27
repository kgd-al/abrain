#include "config.h"

namespace kgd::eshn {

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
}

Config::MutationRates Config::mutationRates =
    normalize<std::string, float>({
                { "add_n",  .5f   },
                { "add_l",  .5f   },
                { "del_n",  .25f  },
                { "del_l",  .25f  },
                { "mut_w", 5.5f   },
                { "mut_f", 2.5f   },
});

Config::FBounds Config::weightBounds {-.3f, -1.f, 1.f, 3.f};

} // end of namespace kgd::eshn
