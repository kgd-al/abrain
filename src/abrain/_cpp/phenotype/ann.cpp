#include <memory>
#include <cassert>
#include <algorithm>

#include "../config.h"
#include "ann.h"

#include <iostream>

namespace kgd::eshn::phenotype {

#ifndef NDEBUG
//#define DEBUG_COMPUTE 1
#endif

#ifndef NDEBUG  // Timing utilities
using timing_clock = std::chrono::high_resolution_clock;
using time_point = std::chrono::time_point<timing_clock>;

static time_point t_now(void) { return timing_clock::now(); }
static ANN::Stats::rep t_diff(time_point start) {
  return std::chrono::duration_cast<ANN::Stats::duration>(t_now() - start).count();
}
#endif

using Output = CPPN::Output;
using Point = es::Point;

bool ANN::empty(bool strict) const {
  return strict ? (stats().hidden == 0) : (stats().edges == 0);
}

bool ANN::perceptron(void) const {
  return (stats().hidden == 0) && stats().edges > 0;
}

ANN ANN::build (const Coordinates &inputs,
                const Coordinates &outputs,
                const genotype::CPPNData &genome) {

#ifndef NDEBUG
//  const auto start_time = t_now();
const auto start_time = timing_clock::now();
#endif

  static const auto& weightRange = Config::annWeightsRange;

  CPPN cppn (genome);
  ANN ann;

  NeuronsMap &neurons = ann._neurons;

  const auto add = [&cppn, &ann] (auto p, auto t) {
    float bias = 0;
    if (t != Neuron::I)
      bias = cppn(p, Point::null(), Output::Bias);
    return ann.addNeuron(p, t, bias);
  };

  uint i = 0;
  ann._inputs.resize(inputs.size());
  ann._ibuffer.resize(inputs.size());
  for (auto &p: inputs) neurons.insert(ann._inputs[i++] = add(p, Neuron::I));

  i = 0;
  ann._outputs.resize(outputs.size());
  ann._obuffer.resize(outputs.size());
  for (auto &p: outputs) neurons.insert(ann._outputs[i++] = add(p, Neuron::O));

  Coordinates hidden;
  evolvable_substrate::Connections connections;
  if (evolvable_substrate::connect(cppn, inputs, outputs,
                                   hidden, connections,
                                   ann._stats.iterations)) {
    for (auto &p: hidden) neurons.insert(add(p, Neuron::H));
    for (auto &c: connections)
      ann.neuronAt(c.to)->addLink(c.weight * weightRange, ann.neuronAt(c.from));
  }

  ann.computeStats();

#ifndef NDEBUG
  ann._stats.time.build = t_diff(start_time);
  ann._stats.time.eval = Stats::rep{0};
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

void ANN::reset(void) {
  for (auto &p: _neurons)   p->reset();
}

uint computeDepth (ANN &ann) {
  struct ReverseNeuron {
    ANN::Neuron &n;
    std::vector<ReverseNeuron*> o;
    ReverseNeuron (ANN::Neuron &n) : n(n) {}
  };

  std::map<Point, ReverseNeuron*> neurons;
  std::set<ReverseNeuron*> next;

  uint hcount = 0, hseen = 0;
  for (const ANN::Neuron::ptr &n: ann.neurons()) {
    auto p = neurons.emplace(std::make_pair(n->pos, new ReverseNeuron(*n)));
    if (n->type == ANN::Neuron::I) next.insert(p.first->second);
    else if (n->type == ANN::Neuron::H) hcount++;
  }

  for (const ANN::Neuron::ptr &n: ann.neurons())
    for (phenotype::ANN::Neuron::Link &l: n->links())
      neurons.at(l.in.lock()->pos)->o.push_back(neurons.at(n->pos));

  std::set<ReverseNeuron*> seen;
  uint depth = 0;
  while (!next.empty()) {
    auto current = next;
    next.clear();

    for (ReverseNeuron *n: current) {
      n->n.depth = depth;
//      std::cerr << n->n.pos << ": " << depth << "\n";
      seen.insert(n);
      if (n->n.type == ANN::Neuron::H)  hseen++;
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

  uint d = 0;
  for (auto &p: neurons) {
    d = std::max(d, p.second->n.depth);
    delete p.second;
  }
  return d;
}

void ANN::computeStats(void) {
  auto &e = _stats.edges = 0;
  float &l = _stats.axons = 0;
  for (const Neuron::ptr &n: _neurons) {
    e += uint(n->links().size());
    for (const Neuron::Link &link: n->links())
      l += (n->pos - link.in.lock()->pos).length();
  }

  _stats.hidden = uint(_neurons.size() - _inputs.size() - _outputs.size());
  _stats.density = _stats.edges;
  if (_stats.hidden == 0) {
    for (Neuron::ptr &n: _inputs)   n->depth = 0;
    for (Neuron::ptr &n: _outputs)  n->depth = 1;
    _stats.depth = 1;
    _stats.density /= _inputs.size() * _outputs.size();

  } else {
    _stats.depth = computeDepth(*this);
    _stats.density /= (_inputs.size() * _stats.hidden + _stats.hidden * _outputs.size());
  }
}

ANN::Neuron::ptr ANN::addNeuron(const Point &p, Neuron::Type t, float bias) {
  return std::make_shared<Neuron>(p, t, bias);
}

void ANN::operator() (const IBuffer &inputs, OBuffer &outputs, uint substeps) {
#ifndef NDEBUG
  const auto start_time = t_now();
#endif

  static const auto &activation =
    phenotype::CPPN::functions.at(Config::activationFunc);
  assert(inputs.size() == _inputs.size());
  assert(outputs.size() == outputs.size());

  for (uint i=0; i<inputs.size(); i++) _inputs[i]->value = inputs[i];

#ifdef DEBUG_COMPUTE
  std::cerr << std::setprecision(std::numeric_limits<float>::max_digits10);
  using utils::operator<<;
  std::cerr << "## Compute step --\n inputs:\t" << inputs << "\n";
#endif

  for (uint s = 0; s < substeps; s++) {
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

  for (uint i=0; i<_outputs.size(); i++)  outputs[i] = _outputs[i]->value;

#ifdef DEBUG_COMPUTE
  std::cerr << "outputs:\t" << outputs << "\n## --\n";
#endif

#ifndef NDEBUG
  _stats.time.eval += t_diff(start_time);
#endif

}

} // end of namespace kgd::eshn::genotype
