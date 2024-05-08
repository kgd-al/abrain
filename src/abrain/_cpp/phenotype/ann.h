#ifndef KGD_ANN_PHENOTYPE_H
#define KGD_ANN_PHENOTYPE_H

#include <vector>

#ifndef NDEBUG
#include <chrono>
#endif

#include "cppn.h"
#include "eshn.h"

namespace kgd::eshn::phenotype {
namespace es = evolvable_substrate;

template <uint DI>
class ANN_t {
public:
  using CPPN = CPPN_ND<DI>;
  using Point = typename CPPN::Point;
  static constexpr auto DIMENSIONS = DI;

  struct Neuron {// : public std::enable_shared_from_this<Neuron> {
    const Point pos;
    enum Type { I, O, H };
    const Type type;

    const float bias;
    float value;

    uint depth = 0;

    // For clustering purposes (bitwise mask)
    using Flags_t = uint;
    Flags_t flags;

    Neuron (const Point &p, Type t, float b)
      : pos(p), type(t), bias(b), value(0), flags(0) {}

    [[nodiscard]] bool isInput () const {   return type == I;  }
    [[nodiscard]] bool isOutput () const {  return type == O; }
    [[nodiscard]] bool isHidden () const {  return type == H; }

    void reset () { value = 0;  }

    using ptr = std::shared_ptr<Neuron>;
    using wptr = std::weak_ptr<Neuron>;
    struct Link {
      float weight;
      wptr in;
    };
    using Links = std::vector<Link>;

    const Links& links () const {   return _ilinks; }
    Links& links () {               return _ilinks; }

    void addLink (float w, wptr n) {    _ilinks.push_back({w,n});  }

    static void assertEqual (const Neuron &lhs, const Neuron &rhs,
                             bool deepcopy);

    static void assertEqual (const Link &lhs, const Link &rhs, bool deepcopy);

  private:
    Links _ilinks;
  };

  struct Stats {
    uint depth;
    uint hidden;
    uint edges;
    float axons;  // total length
    float density;
    uint iterations;
#ifndef NDEBUG
    using duration = std::chrono::duration<double>;
    using rep = duration::rep;
    struct {
      rep build; // Construction time
      rep eval; // Evaluation time
    } time;
#endif
  };
  using NeuronPtr = typename Neuron::ptr;

  ANN_t() = default;

  const auto& neurons () const {  return _neurons;  }
  auto& neurons () {  return _neurons;  }

  const NeuronPtr& neuronAt (const Point &p) const {
    auto it = _neurons.find(p);
    if (it == _neurons.end())
      throw std::invalid_argument("No neuron there");
    return *it;
  }

  struct IBuffer : std::vector<float> {};
  auto& ibuffer () { return _ibuffer; }

  struct OBuffer : std::vector<float> {};
  auto& obuffer () { return _obuffer; }

  void reset ();

  /// TODO Modify with buffer-based eval. Maybe
  /// .. todo:: Modify with buffer-based eval. Maybe
  void operator() (const IBuffer &inputs, OBuffer &outputs, uint substeps = 1);

  [[nodiscard]] bool empty (bool strict = false) const;
  [[nodiscard]] bool perceptron () const;

  void computeStats ();
  const auto& stats () const {
    return _stats;
  }

  // Deepcopy for develop once / use many
  // Not useful yet.
//  void copyInto (ANN &that) const;

  using Coordinates = es::Coordinates_t<DI>;
  static ANN_t build (const Coordinates &inputs,
                    const Coordinates &outputs,
                    const genotype::CPPNData &genome);

private:
  struct NeuronCMP {
    using is_transparent = void;
    bool operator() (const Point &lhs, const Point &rhs) const {
      if (lhs.y() != rhs.y()) return lhs.y() < rhs.y();
      if constexpr (DIMENSIONS == 3)
        if (lhs.z() != rhs.z()) return lhs.z() < rhs.z();
      return lhs.x() < rhs.x();
    }

    bool operator() (const NeuronPtr &lhs, const Point &rhs) const {
      return operator()(lhs->pos, rhs);
    }

    bool operator() (const Point &lhs, const NeuronPtr &rhs) const {
      return operator()(lhs, rhs->pos);
    }

    bool operator() (const NeuronPtr &lhs, const NeuronPtr &rhs) const {
      return operator()(lhs->pos, rhs->pos);
    }
  };
public:
  using NeuronsMap = std::set<NeuronPtr, NeuronCMP>;
private:
  NeuronsMap _neurons;

  std::vector<NeuronPtr> _inputs, _outputs;
  IBuffer _ibuffer;
  OBuffer _obuffer;

  Stats _stats;

  NeuronPtr addNeuron (const Point &p, typename Neuron::Type t, float bias);
};

using ANN2D = ANN_t<2>;
using ANN3D = ANN_t<3>;

} // end of namespace kgd::eshn::phenotype

#endif // KGD_ANN_PHENOTYPE_H
