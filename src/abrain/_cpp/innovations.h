#ifndef KGD_CPPN_GENOTYPE_INNOVATIONS_H
#define KGD_CPPN_GENOTYPE_INNOVATIONS_H

#include "genotype.h"

namespace kgd::eshn::genotype {

class Innovations {
public:
  using ID = CPPNData::ID;
  using Key = std::pair<ID, ID>;
  using Map = std::map<Key, ID>;

  static constexpr auto NOT_FOUND = ID(-1);

  Innovations();
//  Innovations(...); // from an existing genome/population?
  Innovations (const Map &&nodes, const Map &&links,
               const ID &nextNodeId, const ID &nextLinkID);

  void initialize (ID nextNodeID);

  ID link_innovation_id (const ID &from, const ID &to);
  ID node_innovation_id (const ID &from, const ID &to);

  ID add_link_innovation(const ID &from, const ID &to);
  ID add_node_innovation(const ID &from, const ID &to);

  [[nodiscard]] const auto& nodes (void) const { return _nodes; }
  [[nodiscard]] const auto& links (void) const { return _links; }

  [[nodiscard]] const auto& nextNodeID (void) const { return _nextNodeID; }
  [[nodiscard]] const auto& nextLinkID (void) const { return _nextLinkID; }

  [[nodiscard]] Innovations copy() const;

private:
  Map _nodes, _links;
  ID _nextNodeID, _nextLinkID;

  static ID innovation_id (const ID &from, const ID &to, const Map &map);
  static ID add_innovation (const ID &from, const ID &to, Map &map, ID &nextID);
};

} // end of namespace kgd::eshn::genotype

#endif // KGD_CPPN_GENOTYPE_INNOVATIONS_H
