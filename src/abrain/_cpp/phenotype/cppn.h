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

  std::vector<Node_ptr> _inputs, _hidden, _outputs;

public:
  explicit CPPN(const Genotype &genotype);

  using Outputs = std::vector<float>;

  [[nodiscard]] auto obuffer () const { return Outputs(_outputs.size()); }

  // void operator() (Outputs &outputs, float inputs...);
};

template <uint DI>
class CPPN_ND : public CPPN {
public:
  // Specific CPPN for ES-HyperNEAT
  static constexpr auto DIMENSIONS = DI;

  using Point = kgd::eshn::misc::Point_t<DIMENSIONS>;

  explicit CPPN_ND(const Genotype &genotype);

  void operator() (const Point &src, const Point &dst, Outputs &outputs);

  using Output = Config::ESHNOutputs;
  float operator() (const Point &src, const Point &dst, Output o);

  using OutputSubset = std::set<Output>;
  void operator() (const Point &src, const Point &dst, Outputs &outputs,
                   const OutputSubset &oset);

private:
  void pre_evaluation (const Point &src, const Point &dst);
};
using CPPN2D = CPPN_ND<2>;
using CPPN3D = CPPN_ND<3>;

} // end of namespace kgd::eshn::phenotype

#endif // KGD_CPPN_PHENOTYPE_H
