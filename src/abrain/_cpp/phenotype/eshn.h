#ifndef KGD_EVOLV_SUBSTRATE_H
#define KGD_EVOLV_SUBSTRATE_H

#include "cppn.h"

namespace kgd::eshn::evolvable_substrate {

#ifndef NDEBUG
//#define DEBUG_ES 0  // Mute (or leave commented)
//#define DEBUG_ES 1  // Slightly verbose
//#define DEBUG_ES 2
//#define DEBUG_ES 3  // Extremely verbose
#endif

#if DEBUG_ES
//#define DEBUG_QUADTREE 1
#endif

#if DEBUG_QUADTREE
//#define DEBUG_QUADTREE_DIVISION
//#define DEBUG_QUADTREE_PRUNING
struct QuadTreeNode;
namespace quadtree_debug {
void debugGenerateImages (const phenotype::evolvable_substrate::QuadTreeNode &t,
                          const phenotype::ANN::Point &p, bool in);
/// Set where to store debug (a lot of) files
const stdfs::path& debugFilePrefix (const stdfs::path &path = "");
}
#endif

template <unsigned int D>
struct Connection_t {
  using Point = typename misc::Point_t<D>;
  Point from{}, to{};
  float weight{};
#if DEBUG_ES
  friend std::ostream& operator<< (std::ostream &os, const Connection &c) {
    return os << "{ " << c.from << " -> " << c.to << " [" << c.weight << "]}";
  }
#endif
  friend bool operator< (const Connection_t &lhs, const Connection_t &rhs) {
    if (lhs.from != rhs.from) return lhs.from < rhs.from;
    return lhs.to < rhs.to;
  }
};

template <unsigned int D>
using Connections_t = std::set<Connection_t<D>>;

template <unsigned int D>
using Coordinates_t = std::vector<misc::Point_t<D>>;

template <unsigned int D>
bool connect (phenotype::CPPN_ND<D> &cppn,
              const Coordinates_t<D> &inputs, const Coordinates_t<D> &outputs,
              Coordinates_t<D> &hidden, Connections_t<D> &connections,
              unsigned int &iterations);

} // end of namespace kgd::eshn::evolvable_substrate

#endif // KGD_EVOLV_SUBSTRATE_H
