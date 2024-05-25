#ifndef KGD_CPPN_PHENOTYPE_H
#define KGD_CPPN_PHENOTYPE_H

#include <map>
#include <set>
#include <memory>

#include "../config.h"
#include "../genotype.h"
#include "../misc/point.hpp"

namespace kgd::eshn::phenotype {

class CPPN {
  // Generic CPPN
public:
  using Genotype = genotype::CPPNData;

  using FuncID = Genotype::Node::FuncID;
  using Function = float (*) (float);
  using Functions = std::map<FuncID, Function>;
  static const Functions functions;
  using FunctionNames = std::map<Function, FuncID>;
  static const FunctionNames functionToName;

  struct Range { float min, max; };
  static const std::map<FuncID, Range> functionRanges;

protected:
  struct Node_base {
    float data {};
    
    virtual ~Node_base() = default; // LCOVR_EXCL_LINE

    virtual float value () = 0;
  };
  using Node_ptr = std::shared_ptr<Node_base>;
  using Node_wptr = std::weak_ptr<Node_base>;

  struct INode final : Node_base {
    float value () override;
  };

  struct Link {
    float weight;
    Node_wptr node;
  };

  struct FNode final : Node_base {
    float value () override;

    const Function func;

    std::vector<Link> links;

    explicit FNode (const Function f) : func(f) {}
  };

  bool _has_input_bias;
  std::vector<Node_ptr> _inputs, _hidden, _outputs;

public:
  explicit CPPN(const Genotype &genotype);

  struct Buffer : std::vector<float> {
      Buffer () : std::vector<float>() {};
      explicit Buffer (const size_t size) : std::vector<float>(size) {};
  };
  struct IBuffer : Buffer {
      IBuffer () : Buffer() {};
      explicit IBuffer (const size_t size) : Buffer(size) {};
  };
  struct OBuffer : Buffer {
      OBuffer () : Buffer() {};
      explicit OBuffer (const size_t size) : Buffer(size) {};
  };

  [[nodiscard]] auto n_inputs(const bool with_bias=false) const {
    return  _inputs.size() - (1 - with_bias) * _has_input_bias;
  }
  [[nodiscard]] auto n_outputs() const { return _outputs.size(); }
  [[nodiscard]] auto n_hidden() const { return  _hidden.size(); }

  [[nodiscard]] const auto& ibuffer () const { return _ibuffer; }
  [[nodiscard]] const auto& obuffer () const { return _obuffer; }

  void operator() (OBuffer &outputs, const IBuffer &inputs);
  float operator() (unsigned int o, const IBuffer &inputs);

private:
    void pre_evaluation(const IBuffer &inputs);
protected:
    void common_pre_evaluation();

    IBuffer _ibuffer;
    OBuffer _obuffer;
};

template <unsigned int DI>
class CPPN_ND : public CPPN {
public:
  // Specific CPPN for ES-HyperNEAT
  static constexpr auto DIMENSIONS = DI;

  using Point = kgd::eshn::misc::Point_t<DIMENSIONS>;

  explicit CPPN_ND(const Genotype &genotype);

  void operator() (const Point &src, const Point &dst, OBuffer &outputs);

  using Output = Config::ESHNOutputs;
  float operator() (const Point &src, const Point &dst, Output o);

  using OutputSubset = std::set<Output>;
  void operator() (const Point &src, const Point &dst, OBuffer &outputs,
                   const OutputSubset &oset);

private:
  void pre_evaluation (const Point &src, const Point &dst);
};
using CPPN2D = CPPN_ND<2>;
using CPPN3D = CPPN_ND<3>;

} // end of namespace kgd::eshn::phenotype

#endif // KGD_CPPN_PHENOTYPE_H
