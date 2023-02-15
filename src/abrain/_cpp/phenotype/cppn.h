#ifndef KGD_CPPN_PHENOTYPE_H
#define KGD_CPPN_PHENOTYPE_H

#include <map>
#include <set>
#include <memory>

#include "../genotype.h"
#include "../misc/point.hpp"

namespace kgd::eshn::phenotype {

class CPPN {
public:
  using Point = kgd::eshn::misc::Point;
  static constexpr auto DIMENSIONS = Point::DIMENSIONS;
  static constexpr auto INPUTS =  genotype::CPPNData::INPUTS;
  static constexpr auto OUTPUTS = genotype::CPPNData::OUTPUTS;

  using Genotype = kgd::eshn::genotype::CPPNData;

  using FuncID = Genotype::Node::FuncID;
  using Function = float (*) (float);
  using Functions = std::map<FuncID, Function>;
  static const Functions functions;
  using FunctionNames = std::map<CPPN::Function, FuncID>;
  static const FunctionNames functionToName;

  struct Range { float min, max; };
  static const std::map<FuncID, Range> functionRanges;

private:
  struct Node_base {
    float data;
    
    virtual ~Node_base(void) = default; // LCOVR_EXCL_LINE

    virtual float value (void) = 0;
  };
  using Node_ptr = std::shared_ptr<Node_base>;
  using Node_wptr = std::weak_ptr<Node_base>;

  struct INode final : public Node_base {
    float value (void) override;
  };

  struct Link {
    float weight;
    Node_wptr node;
  };

  struct FNode final : public Node_base {
    float value (void) override;

    const Function func;

    std::vector<Link> links;

    FNode (Function f) : func(f) {}
  };

  std::vector<Node_ptr> _inputs, _hidden, _outputs;

public:
  CPPN(const Genotype &genotype);

  static constexpr auto &OUTPUTS_LIST = cppn::CPPN_OUTPUT_LIST;
  using Outputs = std::array<float, OUTPUTS>;

  void operator() (const Point &src, const Point &dst, Outputs &outputs);

  using Output = Genotype::Output;
  float operator() (const Point &src, const Point &dst, Output o);

  using OutputSubset = std::set<Output>;
  void operator() (const Point &src, const Point &dst, Outputs &outputs,
                   const OutputSubset &oset);

private:
  void pre_evaluation (const Point &src, const Point &dst);
};

} // end of namespace kgd::eshn::phenotype

#endif // KGD_CPPN_PHENOTYPE_H
