#ifndef KGD_ANN_PHENOTYPE_H
#define KGD_ANN_PHENOTYPE_H

#include <vector>

#include "cppn.h"
#include "eshn.h"

namespace kgd::eshn::phenotype {
namespace es = evolvable_substrate;

class ANN {
public:
  using Point = es::Point;
  static constexpr auto DIMENSIONS = CPPN::DIMENSIONS;

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

    bool isInput (void) const {   return type == I;  }
    bool isOutput (void) const {  return type == O; }
    bool isHidden (void) const {  return type == H; }

    void reset (void) { value = 0;  }

    using ptr = std::shared_ptr<Neuron>;
    using wptr = std::weak_ptr<Neuron>;
    struct Link {
      float weight;
      wptr in;
    };
    using Links = std::vector<Link>;

    const Links& links (void) const {   return _ilinks; }
    Links& links (void) {               return _ilinks; }

    void addLink (float w, wptr n) {    _ilinks.push_back({w,n});  }

    friend void assertEqual (const Neuron &lhs, const Neuron &rhs,
                             bool deepcopy);

    friend void assertEqual (const Link &lhs, const Link &rhs, bool deepcopy);

  private:
    Links _ilinks;
  };

  struct Stats {
    uint depth;
    uint edges;
    float axons;  // total length
  };

  ANN(void) = default;

  const auto& neurons (void) const {  return _neurons;  }
  auto& neurons (void) {  return _neurons;  }

  const Neuron::ptr& neuronAt (const Point &p) const {
    auto it = _neurons.find(p);
    if (it == _neurons.end())
      throw std::invalid_argument("No neuron there");
    return *it;
  }

  using Inputs = std::vector<float>;
  auto inputsCount (void) const { return _inputs.size(); }
  auto inputs (void) const {  return Inputs(inputsCount(), 0);  }

  using Outputs = Inputs;
  auto outputsCount (void) const { return _outputs.size(); }
  auto outputs (void) const { return Outputs(outputsCount(), 0);  }

//  void reset (void);

  /// TODO Modify with buffer-based eval
  void operator() (const Inputs &inputs, Outputs &outputs, uint substeps);

  bool empty (void) const;

  void computeStats (void);
  const auto& stats (void) const {
    return _stats;
  }

  void copyInto (ANN &that) const;

  using Coordinates = es::Coordinates;
  static ANN build (const Coordinates &inputs,
                    const Coordinates &outputs,
                    const genotype::CPPNData &genome);

private:
  struct NeuronCMP {
    using is_transparent = void;
    bool operator() (const Point &lhs, const Point &rhs) const {
      if (lhs.y() != rhs.y()) return lhs.y() < rhs.y();
      if (lhs.z() != rhs.z()) return lhs.z() < rhs.z();
      return lhs.x() < rhs.x();
    }

    bool operator() (const Neuron::ptr &lhs, const Point &rhs) const {
      return operator()(lhs->pos, rhs);
    }

    bool operator() (const Point &lhs, const Neuron::ptr &rhs) const {
      return operator()(lhs, rhs->pos);
    }

    bool operator() (const Neuron::ptr &lhs, const Neuron::ptr &rhs) const {
      return operator()(lhs->pos, rhs->pos);
    }
  };
public:
  using NeuronsMap = std::set<Neuron::ptr, NeuronCMP>;
private:
  NeuronsMap _neurons;

  std::vector<Neuron::ptr> _inputs, _outputs;

  Stats _stats;

  Neuron::ptr addNeuron (const Point &p, Neuron::Type t, float bias);
};

} // end of namespace kgd::eshn::phenotype

#endif // KGD_ANN_PHENOTYPE_H
