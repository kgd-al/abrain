#include <memory>
#include <cassert>
#include <algorithm>

#include "../config.h"
#include "../misc/utils.hpp"
#include "ann.h"

#include <iostream>

namespace kgd::eshn::phenotype {

#ifndef NDEBUG
//#define DEBUG_COMPUTE 1
#endif

#ifndef NDEBUG  // Timing utilities
using timing_clock = std::chrono::high_resolution_clock;
using time_point = std::chrono::time_point<timing_clock>;

static time_point t_now() { return timing_clock::now(); }

template <typename ANN>
static typename ANN::Stats::rep t_diff(const time_point start) {
  return std::chrono::duration_cast<typename ANN::Stats::duration>(t_now() - start).count();
}
#endif

template <unsigned int DI>
bool ANN_t<DI>::empty(bool strict) const {
  return strict ? (stats().hidden == 0) : (stats().edges == 0);
}

template <unsigned int DI>
bool ANN_t<DI>::perceptron() const {
  return (stats().hidden == 0) && stats().edges > 0;
}

template <unsigned int DI>
ANN_t<DI> ANN_t<DI>::build (
    const Coordinates &inputs,
    const Coordinates &outputs,
    const genotype::CPPNData &genome) {

  if (genome.inputs < 2*DI + genome.bias || 2*DI + genome.bias + 1 < genome.inputs)
    throw std::invalid_argument(utils::mergeToString(
      "Unable to build a ", DI, "D ANN from a genome with ",
      genome.inputs, " inputs. Did you mix it up with the one for the body?"
    ));
  using ANN = ANN_t<DI>;

#ifndef NDEBUG
const auto start_time = timing_clock::now();
#endif

  static const auto& weightRange = Config::annWeightsRange;

  CPPN cppn (genome);

  ANN ann;

  NeuronsMap &neurons = ann._neurons;

  const auto add = [&cppn, &ann] (auto p, auto t) {
    float bias = 0;
    if (t != Neuron::I)
      bias = cppn(p, Point::null(), CPPN::Output::BIAS);
    return ann.addNeuron(p, t, bias);
  };

  Coordinates sortedInputs (inputs);
  std::ranges::sort(sortedInputs.begin(), sortedInputs.end(), NeuronCMP());

  Coordinates sortedOutputs (outputs);
  std::ranges::sort(sortedOutputs.begin(), sortedOutputs.end(), NeuronCMP());

  unsigned int i = 0;
  ann._inputs.resize(inputs.size());
  ann._ibuffer.resize(inputs.size());
  for (auto &p: sortedInputs) neurons.insert(ann._inputs[i++] = add(p, Neuron::I));

  i = 0;
  ann._outputs.resize(outputs.size());
  ann._obuffer.resize(outputs.size());
  for (auto &p: sortedOutputs) neurons.insert(ann._outputs[i++] = add(p, Neuron::O));

  Coordinates hidden;
  evolvable_substrate::Connections_t<DI> connections;
  if (evolvable_substrate::connect(cppn, sortedInputs, sortedOutputs,
                                   hidden, connections,
                                   ann._stats.iterations)) {
    for (auto &p: hidden) neurons.insert(add(p, Neuron::H));
    for (auto &c: connections)
      ann.neuronAt(c.to)->addLink(c.weight * weightRange, ann.neuronAt(c.from));
  }

  ann.computeStats();

#ifndef NDEBUG
  ann._stats.time.build = t_diff<ANN>(start_time);
  ann._stats.time.eval = typename ANN::Stats::rep{0};
#endif

  return ann;
}

// Deepcopy implementation
// Not used for now
//void ANN::copyInto(ANN &that) const {
//  // Copy neurons
//  for (const Neuron::ptr &n: _neurons) {
//    Neuron::ptr n_ = that.addNeuron(n->pos, n->type, n->bias);
//    that._neurons.insert(n_);
//    n_->value = n->value;
//    n_->depth = n->depth;
//    n_->flags = n->flags;
//  }

//  // Generates links
//  for (const Neuron::ptr &n: _neurons) {
//    const Neuron::ptr n_ = that.neuronAt(n->pos);
//    for (const Neuron::Link &l: n->links()) {
//      n_->addLink(l.weight, that.neuronAt(l.in.lock()->pos));
//    }
//  }

//  // Update I/O buffers
//  that._inputs.reserve(_inputs.size());
//  for (const Neuron::ptr &n: _inputs)
//    that._inputs.push_back(that.neuronAt(n->pos));

//  that._outputs.reserve(_outputs.size());
//  for (const Neuron::ptr &n: _outputs)
//    that._outputs.push_back(that.neuronAt(n->pos));

//  // Copy stats
//  that._stats = _stats;
//}

template <unsigned int DI>
void ANN_t<DI>::reset() {
  for (auto &p: _neurons)   p->reset();
}

template <unsigned int DI>
unsigned int ANN_t<DI>::max_hidden_neurons() {
  return std::pow(2, DI*Config::maxDepth);
}

template <typename ANN>
unsigned int computeDepth (ANN &ann) {
  using Neuron = typename ANN::Neuron;
  using NeuronPtr = typename ANN::NeuronPtr;
  using Point = typename ANN::Point;
  struct ReverseNeuron {
    Neuron &n;
    std::vector<ReverseNeuron*> o;
    explicit ReverseNeuron (Neuron &n) : n(n) {}
  };

  std::map<Point, ReverseNeuron*> neurons;
  std::set<ReverseNeuron*> next;

  unsigned int hcount = 0, hseen = 0;
  for (const NeuronPtr &n: ann.neurons()) {
    auto p = neurons.emplace(std::make_pair(n->pos, new ReverseNeuron(*n)));
    if (n->type == Neuron::I) next.insert(p.first->second);
    else if (n->type == Neuron::H) hcount++;
  }

  for (const NeuronPtr &n: ann.neurons())
    for (auto &l: n->links())
      neurons.at(l.in.lock()->pos)->o.push_back(neurons.at(n->pos));

  std::set<ReverseNeuron*> seen;
  unsigned int depth = 0;
  while (!next.empty()) {
    auto current = next;
    next.clear();

    for (ReverseNeuron *n: current) {
      n->n.depth = depth;
//      std::cerr << n->n.pos << ": " << depth << "\n";
      seen.insert(n);
      if (n->n.type == Neuron::H)  hseen++;
      for (ReverseNeuron *o: n->o) next.insert(o);
    }

    decltype(seen) news;
    std::set_difference(next.begin(), next.end(),
                        seen.begin(), seen.end(),
                        std::inserter(news, news.end()));
    next = news;
    assert(hseen == hcount || next.size() > 0);

    depth++;
  }

  unsigned int d = 0;
  for (auto &p: neurons) {
    d = std::max(d, p.second->n.depth);
    delete p.second;
  }
  return d;
}

template <unsigned int DI>
void ANN_t<DI>::computeStats() {
  using Link = typename Neuron::Link;
  auto &e = _stats.edges = 0;
  float &l = _stats.axons = 0;

  float &u = _stats.utility = 0;
  std::set<Point> connected_inputs;

  for (const NeuronPtr &n: _neurons) {
    e += static_cast<unsigned int>(n->links().size());
    for (const Link &link: n->links()) {
      const auto &_n = link.in.lock();
      l += (n->pos - _n->pos).length();

      if (_n->type == Neuron::I)
        connected_inputs.insert(_n->pos);
    }

    if (n->type == Neuron::O && n->links().size() > 0) u++;
  }

  u += connected_inputs.size();
  u /= float(_inputs.size() + _outputs.size());

  _stats.hidden = static_cast<unsigned int>(_neurons.size() - _inputs.size() - _outputs.size());
  _stats.density = _stats.edges / float(max_edges());

  if (_stats.hidden == 0) {
    for (NeuronPtr &n: _inputs)   n->depth = 0;
    for (NeuronPtr &n: _outputs)  n->depth = 1;
    _stats.depth = 1;

  } else
    _stats.depth = computeDepth(*this);
}

template <unsigned int DI>
typename ANN_t<DI>::NeuronPtr
ANN_t<DI>::addNeuron(const Point &p, typename Neuron::Type t, float bias) {
  return std::make_shared<Neuron>(p, t, bias);
}

template <unsigned int DI>
void ANN_t<DI>::operator() (const IBuffer &inputs, OBuffer &outputs, unsigned int substeps) {
#ifndef NDEBUG
  const auto start_time = t_now();
#endif

  static const auto &activation =
    phenotype::CPPN::functions.at(Config::activationFunc);
  assert(inputs.size() == _inputs.size());
  assert(outputs.size() == outputs.size());

  for (unsigned int i=0; i<inputs.size(); i++) _inputs[i]->value = inputs[i];

#ifdef DEBUG_COMPUTE
  std::cerr << std::setprecision(std::numeric_limits<float>::max_digits10);
  using utils::operator<<;
  std::cerr << "## Compute step --\n inputs:\t" << inputs << "\n";
#endif

  for (unsigned int s = 0; s < substeps; s++) {
#ifdef DEBUG_COMPUTE
    std::cerr << "#### Substep " << s+1 << " / " << substeps << "\n";
#endif

    for (const auto &p: _neurons) {
      if (p->isInput()) continue;

      float v = p->bias;
      for (const auto &l: p->links()) {
#if DEBUG_COMPUTE >= 3
        std::cerr << "        i> v = " << v + l.weight * l.in.lock()->value
                  << " = " << v << " + " << l.weight << " * "
                  << l.in.lock()->value << "\n";
#endif

        v += l.weight * l.in.lock()->value;
      }

      p->value = activation(v);
      assert(-1 <= p->value && p->value <= 1);

#if DEBUG_COMPUTE >= 2
      std::cerr << "      <o " << p->pos << ": " << p->value << " = "
                << config::EvolvableSubstrate::activationFunc() << "("
                << v << ")\n";
#endif
    }
  }

  for (unsigned int i=0; i<_outputs.size(); i++)  outputs[i] = _outputs[i]->value;

#ifdef DEBUG_COMPUTE
  std::cerr << "outputs:\t" << outputs << "\n## --\n";
#endif

#ifndef NDEBUG
  _stats.time.eval += t_diff<ANN_t<DI>>(start_time);
#endif

}

template class ANN_t<2>;
template class ANN_t<3>;

} // end of namespace kgd::eshn::genotype
