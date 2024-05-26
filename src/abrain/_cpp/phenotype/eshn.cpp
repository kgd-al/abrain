#include <iostream>
#include <algorithm>
#include <queue>

#include "eshn.h"

#include "ann.h"

namespace kgd::eshn::evolvable_substrate {

template <unsigned int D>
struct ESHN {
  using Point = misc::Point_t<D>;
  using CPPN = phenotype::CPPN_ND<D>;
  using ANN = phenotype::ANN_t<D>;

  using Coordinates = typename ANN::Coordinates;
  using Coordinates_s = std::set<typename Coordinates::value_type>;

  using Connection = Connection_t<D>;
  using Connections = Connections_t<D>;

  using Output = typename CPPN::Output;

  struct QOTreeNode {
    Point center;
    float radius;
    unsigned int level;
    float weight;

    using ptr = std::shared_ptr<QOTreeNode>;
    std::vector<ptr> cs;

    QOTreeNode (const Point &p, const float r, const unsigned int l)
      : center(p), radius(r), level(l), weight(NAN) {}

    QOTreeNode (float x, float y, float r, unsigned int l) requires (D == 2)
      : QOTreeNode({x,y}, r, l) {}

    QOTreeNode (float x, float y, float z, float r, unsigned int l) requires (D == 3)
      : QOTreeNode({x,y,z}, r, l) {}

    [[nodiscard]] float variance () const {
      if (cs.empty()) return 0;
      float mean = 0;
      for (auto &c: cs) mean += c->weight;
      mean /= cs.size();
      float var = 0;
      for (auto &c: cs) var += static_cast<float>(std::pow(c->weight - mean, 2));
      return var / cs.size();
    }

#if DEBUG_ES_QUADTREE
    friend std::ostream& operator<< (std::ostream &os, const QuadTreeNode &n) {
      utils::IndentingOStreambuf indent (os);
      os << "QTN " << n.center << " " << n.radius << " " << n.level << " "
                << n.weight << "\n";
      for (const auto &c: n.cs) os << *c;
      return os;
    }
#endif
  };
  using QOTree = typename QOTreeNode::ptr;

  // =============================================================================
  // Debug streamers

#if DEBUG_ES >= 3  // Only in deep debug
  std::ostream& operator<< (std::ostream &os, const Coordinates_s &c) {
    os << "[";
    if (c.size() >= 1) {
      os << " " << *c.begin() << " ";
      for (auto it = std::next(c.begin()); it != c.end(); ++it) os << *it << " ";
    }
    return os << "]";
  }

  void showConnections(std::ostream &os, const Connections &c, size_t start = 0) {
    auto it = c.begin();
    std::advance(it, start);
    for (; it != c.end(); ++it) os << "\t" << *it << "\n";
  }
#endif

  // =============================================================================

  template <typename... ARGS>
  static QOTree node (ARGS... args) {
    return std::make_shared<QOTreeNode>(args...);
  }

  static QOTree divisionAndInitialisation(CPPN &cppn, const Point &p, bool out) {
    static const auto &initialDepth = Config::initialDepth;
    static const auto &maxDepth = Config::maxDepth;
    static const auto &divThr = Config::divThr;

    QOTree root = node(Point::null(), 1.f, 1);
    std::queue<QOTreeNode*> q;
    q.push(root.get());

    const auto weight = [&cppn] (const Point &p0, const Point &p1) {
      return cppn(p0, p1, Output::WEIGHT);
    };

#if DEBUG_QUADTREE_DIVISION
    std::cout << "divisionAndInitialisation(" << p << ", " << out << ")\n";
#endif

    while (!q.empty()) {
      QOTreeNode &n = *q.front();
      q.pop();

      n.cs.resize(1 << D);
      unsigned int i=0;

      const float hr = .5 * n.radius;
      const float nl = static_cast<float>(n.level) + 1;
      const auto &nc = n.center;
      if constexpr (D == 2) {
        const float cx = nc.x(), cy = nc.y();
        for (const float x: {-1.f, 1.f})
          for (const float y: {-1.f, 1.f})
            n.cs[i++] = node(cx + x * hr, cy + y * hr, hr, nl);

      } else if (D == 3) {
        const float cx = nc.x(), cy = nc.y(), cz = nc.z();
        for (const float x: {-1.f, 1.f})
          for (const float y: {-1.f, 1.f})
            for (const float z: {-1.f, 1.f})
                n.cs[i++] = node(cx + x * hr, cy + y * hr, cz + z * hr, hr, nl);

      }

      for (auto &c: n.cs)
        c->weight = out ? weight(p, c->center) : weight(c->center, p);

#if DEBUG_QUADTREE_DIVISION
      std::string indent (2*n.level, ' ');
      std::cout << indent << n.center << ", r=" << n.radius << ", l=" << n.level
                << ":";
      for (auto &c: n.cs) std::cout << " " << c->weight;
      std::cout << "\n" << indent << "> var = " << n.variance() << "\n";
#endif

      if (n.level < initialDepth || (n.level < maxDepth && n.variance() > divThr))
        for (auto &c: n.cs) q.push(c.get());
    }

#if DEBUG_ES_QUADTREE
    std::cerr << *root;
#endif

#if DEBUG_QUADTREE
    quadtree_debug::debugGenerateImages(*root, p, !out);
#endif

    return root;
  }

  static void pruneAndExtract (
          CPPN &cppn, const Point &p, Connections &con,
          const QOTree &t, bool out) {

    static const auto &varThr = Config::varThr;
    static const auto &bndThr = Config::bndThr;
    static const auto leo = [] (auto &_cppn, auto i, auto o) {
      return static_cast<bool>(_cppn(i, o, Output::LEO));
    };

#if DEBUG_QUADTREE_PRUNING
    if (t->level == 1)  std::cout << "\n---\n";
    utils::IndentingOStreambuf indent (std::cout);
    std::cout << "pruneAndExtract(" << p << ", " << t->center << ", "
              << t->radius << ", " << t->level << ", " << out << ") {\n";
#endif

    for (auto &c: t->cs) {
#if DEBUG_QUADTREE_PRUNING
      utils::IndentingOStreambuf indent1 (std::cout);
      std::cout << "processing " << c->center << "\n";
      utils::IndentingOStreambuf indent2 (std::cout);
#endif

      if (c->variance() >= varThr) {
        // More information at lower resolution -> explore
#if DEBUG_QUADTREE_PRUNING
        std::cout << "a> " << c->variance() << " >= " << varThr
                  << " >> digging\n";
#endif
        pruneAndExtract(cppn, p, con, c, out);

      } else {
        // Not enough information at lower resolution -> test if part of band

        float r = c->radius;
        float bnd = 0;

        float cx = c->center.x(), cy = c->center.y();
        const auto dweight = [&cppn, &p, &c, out] (auto... coords) {
          Point src = out ? p : Point{coords...},
                dst = out ? Point{coords...} : p;
          return std::fabs(c->weight
                           - cppn(src, dst, Output::WEIGHT));
        };

        if constexpr (D == 2) {
          bnd = std::max(
            std::min(dweight(cx-r, cy), dweight(cx+r, cy)),
            std::min(dweight(cx, cy-r), dweight(cx, cy+r))
          );

        } else {
          float cz = c->center.z();
          std::vector<float> bnds {
            std::min(dweight(cx-r, cy, cz), dweight(cx+r, cy, cz)),
            std::min(dweight(cx, cy-r, cz), dweight(cx, cy+r, cz)),
            std::min(dweight(cx, cy, cz-r), dweight(cx, cy, cz+r))
          };
          bnd = *std::ranges::max_element(bnds.begin(), bnds.end());
        }

#if DEBUG_QUADTREE_PRUNING
        std::cout << "b> var = " << c->variance() << ", bnd = "
                  << std::max(std::min(dl, dr), std::min(dt, db))
                  << " = max(min(" << dl << ", " << dr << "), min(" << dt
                  << ", " << db << ")) && leo = "
                  << leoConnection(cppn, out ? p : c->center, out ? c->center : p)
                  << "\n";
#endif

        if (bnd > bndThr
            && leo(cppn, out ? p : c->center, out ? c->center : p)
            && c->weight != 0) {
          con.insert({
            out ? p : c->center, out ? c->center : p, c->weight
          });
#if DEBUG_QUADTREE_PRUNING
          std::cout << " < created " << (out ? p : c->center) << " -> "
                    << (out ? c->center : p) << " [" << c->weight << "]\n";
#endif
        }
      }
    }

#if DEBUG_QUADTREE_PRUNING
    std::cout << "}\n";
#endif
  }

  using Type = typename ANN::Neuron::Type;
  struct L;
  struct N {
      Point p;
      Type t;
      std::vector<L> i, o;
      explicit N (const Point &p, Type t = Type::H) : p(p), t(t) {}
  };
  struct L {
      N *n;
      float w;
      explicit L (N *n, const float w) : n(n), w(w) {}
  };
  struct CMP {
      using is_transparent [[maybe_unused]] = void;
      bool operator() (const N *lhs, const Point &rhs) const {
        return lhs->p < rhs;
      }
      bool operator() (const Point &lhs, const N *rhs) const {
        return lhs < rhs->p;
      }
      bool operator() (const N *lhs, const N *rhs) const {
        return lhs->p < rhs->p;
      }
  };

  static void removeUnconnectedNeurons (
      const Coordinates &inputs, const Coordinates &outputs,
      Coordinates_s &shidden, Connections &connections) {

    std::set<N*, CMP> nodes, inodes, onodes;
    for (const Point &p: inputs)
      nodes.insert(*inodes.insert(new N(p, Type::I)).first);
    for (const Point &p: outputs)
      nodes.insert(*onodes.insert(new N(p, Type::O)).first);

    const auto getOrCreate = [&nodes] (const Point &p) {
      auto it = nodes.find(p);
      if (it != nodes.end())
        return *it;
      return *nodes.insert(new N(p)).first;
    };
    for (const Connection &c: connections) {
      N *i = getOrCreate(c.from), *o = getOrCreate(c.to);
      i->o.push_back(L(o, c.weight));
      o->i.push_back(L(i, c.weight));
    }

#if DEBUG_ES >= 2
    std::cerr << "\ninodes:\n";
    for (const auto &n: inodes) std::cerr << "\t" << n->p << "\n";
    std::cerr << "\nonodes:\n";
    for (const auto &n: onodes) std::cerr << "\t" << n->p << "\n";
    std::cerr << "\nconnections:\n";
    for (const auto &c: connections)
      std::cerr << "\t" << c.from << " -> " << c.to << "\n";
    std::cerr << "\n";
#endif

    const auto breadthFirstSearch =
        [] (const auto &src, auto &set, auto field) {
          std::queue<N*> q;
          typename std::remove_reference_t<decltype(set)> seen;
          for (const auto &n: src)  q.push(n), seen.insert(n);
          while (!q.empty()) {
            N *n = q.front();
            q.pop();

            for (auto &l: n->*field) {
              if (N *n_ = l.n; seen.find(n_) == seen.end()) {
                if (n_->t == Type::H) set.insert(n_);
                seen.insert(n_);
                q.push(n_);
              }
            }
          }
    };

    std::set<N*, CMP> iseen, oseen;
    breadthFirstSearch(inodes, iseen, &N::o);
    breadthFirstSearch(onodes, oseen, &N::i);

#if DEBUG_ES >= 2
    std::cerr << "hidden nodes:\n\tiseen:\n";
    for (const auto *n: iseen)  std::cerr << "\t\t" << n->p << "\n";
    std::cerr << "\n\toseen:\n";
    for (const auto *n: oseen)  std::cerr << "\t\t" << n->p << "\n";
    std::cerr << "\n";
#endif

    CMP cmp;
    std::vector<N*> hiddenNodes;
    std::set_intersection(iseen.begin(), iseen.end(), oseen.begin(), oseen.end(),
                          std::back_inserter(hiddenNodes), cmp);

    connections.clear();
    for (const N *n: hiddenNodes) {
#if DEBUG_ES >= 3
      std::cerr << "\t" << n->p << "\n";
#endif
      shidden.insert(n->p);
      for (const L &l: n->i)  connections.insert({l.n->p, n->p, l.w});
      for (const L &l: n->o)
        if (l.n->t == Type::O)
          connections.insert({n->p, l.n->p, l.w});
    }

    for (auto it=nodes.begin(); it!=nodes.end();) {
      delete *it;
      it = nodes.erase(it);
    }
  }

  /// Collect new hidden nodes and connections
  static void collect (const Connections &newConnections, Connections &connections,
                       Coordinates_s &hiddens, Coordinates_s &newHiddens) {
    for (auto &c: newConnections) {
      auto r = hiddens.insert(c.to);
      if (r.second) newHiddens.insert(c.to);
    }
    connections.insert(newConnections.begin(), newConnections.end());
  }

  /// Query for direct input-output connections
  static void generatePerceptron(CPPN &cppn,
                                 const Coordinates &inputs, const Coordinates &outputs,
                                 Connections &connections) {

    typename CPPN::OutputSubset wl {{ Output::WEIGHT, Output::LEO }};
    auto res = cppn.obuffer();

    for (const Point &i: inputs) {
      for (const Point &o: outputs) {
        cppn(i, o, res, wl);
        if (res[static_cast<unsigned int>(Output::LEO)])
          connections.insert({i, o, res[static_cast<unsigned int>(Output::WEIGHT)]});
      }
    }
  }
};

// Triggered by duplicate coordinates
template <unsigned int D>
std::ostream& operator<< (std::ostream &os, const Coordinates_t<D> &c) {
  os << "[";
  if (!c.empty()) os << " " << c[0] << " ";
  for (unsigned int i=1; i<c.size(); i++) os << c[i] << " ";
  return os << "]";
}

template <unsigned int D>
bool connect (phenotype::CPPN_ND<D> &cppn,
              const Coordinates_t<D> &inputs, const Coordinates_t<D> &outputs,
              Coordinates_t<D> &hidden, Connections_t<D> &connections, unsigned int &iterations) {

  using E = ESHN<D>;
  using Coordinates_s = typename E::Coordinates_s;
  using Connections = typename E::Connections;
  using Point = typename E::Point;

  static const auto &max_iterations = Config::iterations;

  Coordinates_s sio;  // All fixed positions
  for (const auto &vec: {inputs, outputs}) {
    for (Point p: vec) {
      if (const auto r = sio.insert(p); !r.second) {
        std::cerr << "inputs: " << inputs << "\noutputs: " << outputs
                  << std::endl;
        throw std::invalid_argument("Unable to insert duplicate coordinate ");
      }
    }
  }

#if DEBUG_ES
  std::ostringstream oss;
  oss << "\n## --\nStarting evolvable substrate instantiation\n";
  unsigned int n_hidden = 0, n_connections = 0;
#endif

  Coordinates_s shidden;

  for (const Point &p: inputs) {
    Connections tmpConnections;
    auto t = E::divisionAndInitialisation(cppn, p, true);
    E::pruneAndExtract(cppn, p, tmpConnections, t, true);

    Coordinates_s newHiddens;
    E::collect(tmpConnections, connections, shidden, newHiddens);
  }

#if DEBUG_ES
  oss << "[I -> H] found " << shidden.size() - n_hidden << " hidden neurons";
#if DEBUG_ES >= 3
  oss << "\n\t" << shidden << "\n";
#endif
  oss << " and " << connections.size() - n_connections << " connections";
#if DEBUG_ES >= 3
  oss << "\n";
  showConnections(oss, connections, n_connections);
  oss << "\n";
#endif
  n_hidden = shidden.size();
  n_connections = connections.size();
  oss << "\n";
#endif

  bool converged = false;
  Coordinates_s unexploredHidden = shidden;
  for (iterations = 0; iterations<max_iterations && !converged; iterations++) {

    Coordinates_s newHiddens;
    for (const Point &p: unexploredHidden) {
      Connections tmpConnections;
      auto t = E::divisionAndInitialisation(cppn, p, true);
      E::pruneAndExtract(cppn, p, tmpConnections, t, true);
      E::collect(tmpConnections, connections, shidden, newHiddens);
    }

    unexploredHidden = newHiddens;

#if DEBUG_ES
  oss << "[H -> H] found " << shidden.size() - n_hidden
      << " hidden neurons (" << unexploredHidden.size() << " to explore)";
#if DEBUG_ES >= 3
  oss << "\n\t" << unexploredHidden << "\n";
#endif
  oss << " and " << connections.size() - n_connections << " connections";
#if DEBUG_ES >= 3
  oss << "\n";
  showConnections(oss, connections, n_connections);
  oss << "\n";
#endif
  n_hidden = shidden.size();
  n_connections = connections.size();
  oss << "\n";
#endif

    converged = unexploredHidden.empty();
#if DEBUG_ES
    if (converged)
      oss << "\t> Premature convergence at iteration " << i << "\n";
#endif
  }

  for (const Point &p: outputs) {
    Connections tmpConnections;
    auto t = E::divisionAndInitialisation(cppn, p, false);
    E::pruneAndExtract(cppn, p, tmpConnections, t, false);
    connections.insert(tmpConnections.begin(), tmpConnections.end());
  }

#if DEBUG_ES
  oss << "[H -> O] found " << connections.size() - n_connections
      << " connections";
#if DEBUG_ES >= 3
  oss << "\n";
  showConnections(oss, connections, n_connections);
  oss << "\n";
#endif
  oss << "\n";
#endif

  Coordinates_s shidden2;
  E::removeUnconnectedNeurons(inputs, outputs, shidden2, connections);

#if DEBUG_ES
  oss << "[Filtrd] total " << shidden2.size() << " hidden neurons";
#if DEBUG_ES >= 3
  oss << "\n\t" << shidden2 << "\n";
#endif
  oss << " and " << connections.size() << " connections";
#if DEBUG_ES >= 3
  oss << "\n";
  showConnections(oss, connections);
  oss << "\n";
#endif
#endif

  std::copy(shidden2.begin(), shidden2.end(), std::back_inserter(hidden));

  if (hidden.empty() && Config::allowPerceptrons)
    E::generatePerceptron(cppn, inputs, outputs, connections);

#if DEBUG_ES
  std::cerr << oss.str() << std::endl;
#endif

  return true;
}

using namespace phenotype;
template struct ESHN<2>;
template struct ESHN<3>;

template bool connect<2>(
        phenotype::CPPN2D &cppn,
        const Coordinates_t<2> &inputs,
        const Coordinates_t<2> &outputs,
        Coordinates_t<2> &hidden,
        Connections_t<2> &connections, unsigned int &iterations);

template bool connect<3>(
        phenotype::CPPN3D &cppn,
        const Coordinates_t<3> &inputs,
        const Coordinates_t<3> &outputs,
        Coordinates_t<3> &hidden,
        Connections_t<3> &connections, unsigned int &iterations);

} // end of namespace evolvable substrate
