#include <cmath>
#include <cassert>

#include <iostream>

#include "cppn.h"
#include "../config.h"

#ifndef NDEBUG
//#define DEBUG_CPPN
#include <iomanip>
#endif

#ifdef DEBUG_CPPN
#include <iomanip>

namespace utils { // Contains debugging tools
std::ostream& operator<< (std::ostream &os,
                         const kgd::eshn::phenotype::CPPN::Buffer &buffer) {
  os << "[ " << buffer[0];
  for (unsigned int i=1; i<buffer.size(); i++) os << " " << buffer[i];
  return os << "]";
}

/// Manages indentation for provided ostream
/// \author James Kanze @ https://stackoverflow.com/a/9600752
class IndentingOStreambuf : public std::streambuf {
  static constexpr unsigned int DEFAULT_INDENT = 2;   ///< Default indenting value

  std::ostream*       _owner;   ///< Associated ostream
  std::streambuf*     _buffer;  ///< Associated buffer
  bool                _isAtStartOfLine; ///< Whether to insert indentation

  const std::string   _indent;  ///< Indentation value

protected:
  /// Overrides std::basic_streambuf::overflow to insert indentation at line start
  int overflow (int ch) override {
    if (_isAtStartOfLine && ch != '\n')
      _buffer->sputn(_indent.data(), _indent.size());
    _isAtStartOfLine = (ch == '\n');
    return _buffer->sputc(ch);
  }

public:
  /// Creates a proxy buffer managing indentation level
  explicit IndentingOStreambuf(std::ostream& dest,
                               unsigned int spaces = DEFAULT_INDENT)
    : _owner(&dest), _buffer(dest.rdbuf()),
      _isAtStartOfLine(true),
      _indent(spaces, ' ' ) { _owner->rdbuf( this );  }

  /// Returns control of the buffer to its owner
  virtual ~IndentingOStreambuf(void) { _owner->rdbuf(_buffer); }
};

} // end of namespace utils
#endif

namespace kgd::eshn::phenotype {
using CPPNData = kgd::eshn::genotype::CPPNData;


// =============================================================================

// -- Ugly solution to numerical non-determinism
// Divergences observed on local/remote machines can be solved by requesting
//  value in double precision (accurate within 1ULP?) and truncating back to
//  float. Computationally more expensive but ensures bitwise identical CPPN/ANN

float fd_exp(float x) {
  return static_cast<float>(static_cast<double(*)(double)>(std::exp)(x));
}

float fd_sin(float x) {
  return static_cast<float>(static_cast<double(*)(double)>(std::sin)(x));
}

#define KGD_EXP fd_exp
#define KGD_EXP_STR "dexp"

// =============================================================================

float ssgn(float x) {
  static constexpr float a = 1;
  return x < -a ? KGD_EXP(-(x+a)*(x+a))-1
                : x > a ? 1 - KGD_EXP(-(x-a)*(x-a))
                        : 0;
}

#define F(NAME, BODY) \
 { NAME, [] (float x) -> float { return BODY; } }
const std::map<CPPNData::Node::FuncID,
               CPPN::Function> CPPN::functions {

  F(  "id", x), // Identity
  F( "abs", std::fabs(x)),  // Absolute value
  F( "sin", fd_sin(2.f*x)), // Sinusoidal
  F("step", x <= 0.f ? 0.f : 1.f),  // Step function
  F("gaus", KGD_EXP(-6.25f*x*x)), // Gaussian function
  F("ssgm", 1.f / (1.f + KGD_EXP(-4.9f*x))), // Soft sigmoid
  F("bsgm", 2.f / (1.f + KGD_EXP(-4.9f*x)) - 1.f),  // Bimodal sigmoid

  // Custom-made activation function (with quiescent state)
  F("ssgn", ssgn(x)), // Activation function
};
#undef F

template <typename K, typename V>
std::map<V, K> reverse (const std::map<K, V> &m) {
  std::map<V, K> m_;
  for (const auto &p: m) m_.emplace(p.second, p.first);
  return m_;
} // LCOV_EXCL_LINE

const std::map<CPPN::Function, CPPNData::Node::FuncID>
  CPPN::functionToName = reverse(CPPN::functions);

#define F(NAME, MIN, MAX) { NAME, { MIN, MAX }}
const std::map<CPPNData::Node::FuncID,
               CPPN::Range> CPPN::functionRanges {

  F( "abs",  0, 1),
  F("gaus",  0, 1),
  F(  "id", -1, 1),
  F("ssgm",  0, 1),
  F("bsgm", -1, 1),
  F( "sin", -1, 1),
  F("step",  0, 1),

  F("ssgn", -1, 1),

//  F("kact", -1, 1), // Not really (min value ~ .278)
};
#undef F

// =============================================================================


CPPN::CPPN (const CPPNData &genotype) {
  using NID = unsigned int;//CPPNData::Node::ID;
  const auto NI = genotype.inputs;
  const auto NO = genotype.outputs;
  const auto NH = genotype.nodes.size() - NO;

  auto fnode = [] (const CPPNData::Node::FuncID &fid) {
    return std::make_shared<FNode>(functions.at(fid));
  };

  std::map<NID, Node_ptr> nodes;
#ifdef DEBUG_CPPN
  std::cerr << "Building CPPN with " << NI << " inputs, " << NO << " outputs,"
            << " and " << NH << " internal nodes" << std::endl;
#endif

  _has_input_bias = genotype.bias;

  _inputs.resize(NI);
  _outputs.resize(NO);
  _hidden.resize(NH);

  _ibuffer = IBuffer(n_inputs(false));
  _obuffer = OBuffer(NO);

  for (NID i=0; i<NI; i++) {
#ifdef DEBUG_CPPN
    std::cerr << "(I) " << NID(i) << " " << i << std::endl;
#endif
    nodes[i] = _inputs[i] = std::make_shared<INode>();
  }

  for (NID i=0; i<NO; i++) {
    const auto [id, func] = genotype.nodes[i];
#ifdef DEBUG_CPPN
    std::cerr << "(O) " << i+NI << " " << i << " " << func << std::endl;
#endif
    nodes[id] = _outputs[i] = fnode(func);
  }

  for (NID i=NO; i<genotype.nodes.size(); i++) {
    const auto [id, func] = genotype.nodes[i];
#ifdef DEBUG_CPPN
    std::cerr << "(H) " << id << " " << i << " " << func << std::endl;
#endif
    nodes[id] = _hidden[i-NO] = fnode(func);
  }

  for (const auto &l_g: genotype.links) {
    auto &n = dynamic_cast<FNode&>(*nodes.at(l_g.dst));
    n.links.push_back({l_g.weight, nodes.at(l_g.src)});
  }

#ifdef DEBUG_CPPN
  unsigned int i=0;
  std::map<Node_ptr, unsigned int> map;
  printf("Built CPPN:\n");
  for (const auto &v: {_inputs, _outputs, _hidden})
    for (const Node_ptr &n: v)
      map[n] = i++;

  for (const auto &v: {_hidden, _outputs}) {
    for (const Node_ptr &n: v) {
      FNode &fn = *static_cast<FNode*>(n.get());
      printf("\t[%d]\n", map.at(n));
      std::string fname (functionToName.at(fn.func));
      printf("\t\t%s\n", fname.c_str());
      for (const Link &l: fn.links)
        printf("\t\t[%d]\t%g %a\n", map.at(l.node.lock()), l.weight, l.weight);
    }
  }
  printf("--\n");
#endif
}

float CPPN::INode::value () {
#ifdef DEBUG_CPPN
  utils::IndentingOStreambuf indent (std::cout);
  std::cout << "I: " << data << std::endl;
#endif
  return data;
}

float CPPN::FNode::value () {
#ifdef DEBUG_CPPN
  utils::IndentingOStreambuf indent (std::cout);
  std::cout << "F:\n";
#endif
  if (std::isnan(data)) {
    data = 0.f;
    for (Link &l: links)
      data += l.weight * l.node.lock()->value();

#ifdef DEBUG_CPPN
    auto val = func(data);
    std::cout << val << " = " << functionToName.at(func)
              << "(" << data << ")\n";
    data = val;
#else
    data = func(data);
#endif

#ifdef DEBUG_CPPN
  } else
    std::cout << data << "\n";
#else
  }
#endif
  return data;
}

#ifdef DEBUG_CPPN
std::ostream& operator<< (std::ostream &os, const std::vector<float> &v) {
  os << "[";
  for (float f: v)  os << " " << f;
  return os << " ]";
}
#endif

void CPPN::pre_evaluation(const IBuffer &inputs) {
  if (inputs.size() != n_inputs())
    throw std::runtime_error("Invalid number of inputs");
  for (unsigned int i=0; i<_inputs.size(); i++) _inputs[i]->data = inputs[i];
  common_pre_evaluation();
}

void CPPN::common_pre_evaluation() {
  if (_has_input_bias)
    _inputs.back()->data = 1;

  for (const auto &n: _hidden)  n->data = NAN;
  for (const auto &n: _outputs)  n->data = NAN;

#ifdef DEBUG_CPPN
  utils::IndentingOStreambuf indent (std::cout);
  std::cout << "compute step\n\tInputs:"
            << std::setprecision(std::numeric_limits<float>::max_digits10);
  for (auto &i: _inputs) std::cout << " " << i->data;
  std::cout << "\n";
#endif
}

void CPPN::operator() (OBuffer &outputs, const IBuffer &inputs) {
  pre_evaluation(inputs);
  for (unsigned int i=0; i<outputs.size(); i++)
    outputs[i] = _outputs[i]->value();
}

float CPPN::operator() (unsigned int o, const IBuffer &inputs) {
  pre_evaluation(inputs);
  return _outputs[o]->value();
}


// =============================================================================

template <unsigned int DI>
CPPN_ND<DI>::CPPN_ND(const Genotype &genotype) : CPPN(genotype) {}
template CPPN_ND<2>::CPPN_ND(const Genotype &genotype);
template CPPN_ND<3>::CPPN_ND(const Genotype &genotype);

template <unsigned int DI>
void CPPN_ND<DI>::pre_evaluation(const CPPN_ND<DI>::Point &src,
                                 const CPPN_ND<DI>::Point &dst) {
  static constexpr auto N = DIMENSIONS;
  const auto I = n_inputs(true);
  for (unsigned int i=0; i<N; i++)  _inputs[i]->data = src.get(i);
  for (unsigned int i=0; i<N; i++)  _inputs[i+N]->data = dst.get(i);

  static const auto norm = static_cast<float>(2*std::sqrt(2));
  if (I - static_cast<int>(_has_input_bias) > 2*N)
    _inputs[2*N]->data = (src - dst).length() / norm;

  common_pre_evaluation();
}

template <unsigned int DI>
void CPPN_ND<DI>::operator() (const Point &src, const Point &dst, OBuffer &outputs) {
  assert(outputs.size() == _outputs.size());

  pre_evaluation(src, dst);
  for (unsigned int i=0; i<outputs.size(); i++) outputs[i] = _outputs[i]->value();

#ifdef DEBUG_CPPN
  using utils::operator<<;
  std::cout << outputs << "\n" << std::endl;
#endif
}
template void CPPN_ND<2>::operator() (const Point &src, const Point &dst, OBuffer &outputs);
template void CPPN_ND<3>::operator() (const Point &src, const Point &dst, OBuffer &outputs);

template <unsigned int DI>
void CPPN_ND<DI>::operator() (
    const Point &src, const Point &dst,
    OBuffer &outputs,
    const OutputSubset &oset) {
  assert(outputs.size() == _outputs.size());
  assert(oset.size() <= _outputs.size());

  pre_evaluation(src, dst);
  for (const auto o: oset) outputs[o] = _outputs[o]->value();

#ifdef DEBUG_CPPN
  using utils::operator<<;
  std::cout << outputs << "\n" << std::endl;
#endif
}
template void CPPN_ND<2>::operator() (const Point &src, const Point &dst, OBuffer &outputs, const OutputSubset &oset);
template void CPPN_ND<3>::operator() (const Point &src, const Point &dst, OBuffer &outputs, const OutputSubset &oset);


// Hopefully will get rid of i686 float c++ -> python transfer errors
#if __i386__
#pragma GCC push_options
#pragma GCC optimize ("O0")
#endif
template <unsigned int DI>
float CPPN_ND<DI>::operator() (const Point &src, const Point &dst, const Output o) {
  pre_evaluation(src, dst);
    
  return _outputs[o]->value();
}
template float CPPN_ND<2>::operator() (const Point &src, const Point &dst, Output o);
template float CPPN_ND<3>::operator() (const Point &src, const Point &dst, Output o);
#if __i386__
#pragma GCC pop_options
#endif

} // end of namespace kgd::eshn::phenotype
