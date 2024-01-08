#include <iostream>
#include <algorithm>
#include <queue>

#include "eshn.h"

#include "../config.h"
#include "ann.h"

namespace kgd::eshn::evolvable_substrate {

using ANN = phenotype::ANN;
using Point = ANN::Point;
using Coordinates = phenotype::ANN::Coordinates;
using Coordinates_s = std::set<Coordinates::value_type>;

using Output = CPPN::Output;

struct QOTreeNode {
  Point center;
  float radius;
  uint level;
  float weight;

  using ptr = std::shared_ptr<QOTreeNode>;
  std::vector<ptr> cs;

  QOTreeNode (const Point &p, float r, uint l)
    : center(p), radius(r), level(l), weight(NAN) {}

#if ESHN_SUBSTRATE_DIMENSION == 2
  QOTreeNode (float x, float y, float r, uint l)
    : QOTreeNode({x,y}, r, l) {}
#elif ESHN_SUBSTRATE_DIMENSION == 3
  QOTreeNode (float x, float y, float z, float r, uint l)
    : QOTreeNode({x,y,z}, r, l) {}
#endif

  float variance (void) const {
    if (cs.empty()) return 0;
    float mean = 0;
    for (auto &c: cs) mean += c->weight;
    mean /= cs.size();
    float var = 0;
    for (auto &c: cs) var += float(std::pow(c->weight - mean, 2));
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
using QOTree = QOTreeNode::ptr;

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

// Triggered by duplicate coordinates
std::ostream& operator<< (std::ostream &os, const Coordinates &c) {
  os << "[";
  if (c.size() >= 1) os << " " << c[0] << " ";
  for (uint i=1; i<c.size(); i++) os << c[i] << " ";
  return os << "]";
}

// =============================================================================

template <typename... ARGS>
QOTreeNode::ptr node (ARGS... args) {
  return std::make_shared<QOTreeNode>(args...);
}

QOTree divisionAndInitialisation(CPPN &cppn, const Point &p, bool out) {
  static const auto &initialDepth = Config::initialDepth;
  static const auto &maxDepth = Config::maxDepth;
  static const auto &divThr = Config::divThr;

  QOTree root = node(Point::null(), 1.f, 1);
  std::queue<QOTreeNode*> q;
  q.push(root.get());

  const auto weight = [&cppn] (const Point &p0, const Point &p1) {
    return cppn(p0, p1, Output::Weight);
  };

#if DEBUG_QUADTREE_DIVISION
  std::cout << "divisionAndInitialisation(" << p << ", " << out << ")\n";
#endif

  while (!q.empty()) {
    QOTreeNode &n = *q.front();
    q.pop();

    float cx = n.center.x(), cy = n.center.y();
#if ESHN_SUBSTRATE_DIMENSION == 3
    float cz = n.center.z();
#endif
    float hr = .5 * n.radius;
    float nl = float(n.level) + 1;

    n.cs.resize(1 << ESHN_SUBSTRATE_DIMENSION);
    uint i=0;
    for (int x: {-1,1})
      for (int y: {-1, 1})
#if ESHN_SUBSTRATE_DIMENSION == 2
        n.cs[i++] = node(cx + x * hr, cy + y * hr, hr, nl);
#elif ESHN_SUBSTRATE_DIMENSION == 3
        for (int z: {-1,1})
          n.cs[i++] = node(cx + x * hr, cy + y * hr, cz + z * hr, hr, nl);
#endif

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

void pruneAndExtract (CPPN &cppn, const Point &p, Connections &con,
                      const QOTree &t, bool out) {

  static const auto &varThr = Config::varThr;
  static const auto &bndThr = Config::bndThr;
  static const auto leo = [] (auto &cppn, auto i, auto o) {
    return (bool)cppn(i, o, Output::LEO);
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
                         - cppn(src, dst, Output::Weight));
      };


#if ESHN_SUBSTRATE_DIMENSION == 2
      bnd = std::max(
        std::min(dweight(cx-r, cy), dweight(cx+r, cy)),
        std::min(dweight(cx, cy-r), dweight(cx, cy+r))
      );

#elif ESHN_SUBSTRATE_DIMENSION == 3
      float cz = c->center.z();
      std::vector<float> bnds {
        std::min(dweight(cx-r, cy, cz), dweight(cx+r, cy, cz)),
        std::min(dweight(cx, cy-r, cz), dweight(cx, cy+r, cz)),
        std::min(dweight(cx, cy, cz-r), dweight(cx, cy, cz+r))
      };
      bnd = *std::max_element(bnds.begin(), bnds.end());

#endif

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

void removeUnconnectedNeurons (const Coordinates &inputs,
                               const Coordinates &outputs,
                               Coordinates_s &shidden,
                               Connections &connections) {
  using Type = ANN::Neuron::Type;
  struct L;
  struct N {
    Point p;
    Type t;
    std::vector<L> i, o;
    N (const Point &p, Type t = Type::H) : p(p), t(t) {}
  };
  struct L {
    N *n;
    float w;
  };
  struct CMP {
    using is_transparent = void;
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

  std::set<N*, CMP> nodes, inodes, onodes;
  for (const Point &p: inputs)
    nodes.insert(*inodes.insert(new N(p, Type::I)).first);
  for (const Point &p: outputs)
    nodes.insert(*onodes.insert(new N(p, Type::O)).first);

  const auto getOrCreate = [&nodes] (const Point &p) {
    auto it = nodes.find(p);
    if (it != nodes.end())  return *it;
    else                    return *nodes.insert(new N(p)).first;
  };
  for (const Connection &c: connections) {
    N *i = getOrCreate(c.from), *o = getOrCreate(c.to);
    i->o.push_back({o,c.weight});
    o->i.push_back({i,c.weight});
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
        N *n_ = l.n;
        if (seen.find(n_) == seen.end()) {
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
void collect (const Connections &newConnections, Connections &connections,
              Coordinates_s &hiddens, Coordinates_s &newHiddens) {
  for (auto &c: newConnections) {
    auto r = hiddens.insert(c.to);
    if (r.second) newHiddens.insert(c.to);
  }
  connections.insert(newConnections.begin(), newConnections.end());
}

/// Query for direct input-output connections
void generatePerceptron(CPPN &cppn,
                        const Coordinates &inputs, const Coordinates &outputs,
                        Connections &connections) {

  CPPN::OutputSubset wl {{ Output::Weight, Output::LEO }};
  CPPN::Outputs res;

  for (const Point &i: inputs) {
    for (const Point &o: outputs) {
      cppn(i, o, res, wl);
      if (res[uint(Output::LEO)])
        connections.insert({i, o, res[uint(Output::Weight)]});
    }
  }
}

bool connect (CPPN &cppn,
              const Coordinates &inputs, const Coordinates &outputs,
              Coordinates &hidden, Connections &connections, uint &iterations) {

  static const auto &max_iterations = Config::iterations;

  Coordinates_s sio;  // All fixed positions
  for (const auto &vec: {inputs, outputs}) {
    for (Point p: vec) {
      auto r = sio.insert(p);
      if (!r.second) {
        std::cerr << "inputs: " << inputs << "\noutputs: " << outputs
                  << std::endl;
        throw std::invalid_argument("Unable to insert duplicate coordinate ");
      }
    }
  }

#if DEBUG_ES
  std::ostringstream oss;
  oss << "\n## --\nStarting evolvable substrate instantiation\n";
  uint n_hidden = 0, n_connections = 0;
#endif

  Coordinates_s shidden;

  for (const Point &p: inputs) {
    Connections tmpConnections;
    auto t = divisionAndInitialisation(cppn, p, true);
    pruneAndExtract(cppn, p, tmpConnections, t, true);

    Coordinates_s newHiddens;
    collect(tmpConnections, connections, shidden, newHiddens);
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
      auto t = divisionAndInitialisation(cppn, p, true);
      pruneAndExtract(cppn, p, tmpConnections, t, true);
      collect(tmpConnections, connections, shidden, newHiddens);
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
    auto t = divisionAndInitialisation(cppn, p, false);
    pruneAndExtract(cppn, p, tmpConnections, t, false);
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
  removeUnconnectedNeurons(inputs, outputs, shidden2, connections);

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
    generatePerceptron(cppn, inputs, outputs, connections);

#if DEBUG_ES
  std::cerr << oss.str() << std::endl;
#endif

  return true;
}

} // end of namespace evolvable substrate
