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
//  Innovations(...); // src an existing genome/population?
  Innovations (const Map &&nodes, const Map &&links,
               const ID &nextNodeId, const ID &nextLinkID);

  void initialize (ID nextNodeID);

  ID link_id (const ID &src, const ID &dst) const;
  ID node_id (const ID &src, const ID &dst) const;

  ID get_link_id(const ID &src, const ID &dst);
  ID get_node_id(const ID &src, const ID &dst);

  ID new_link_id(const ID &src, const ID &dst);
  ID new_node_id(const ID &src, const ID &dst);

  [[nodiscard]] const auto& nodes (void) const { return _nodes; }
  [[nodiscard]] const auto& links (void) const { return _links; }

  [[nodiscard]] const auto& nextNodeID (void) const { return _nextNodeID; }
  [[nodiscard]] const auto& nextLinkID (void) const { return _nextLinkID; }

  [[nodiscard]] Innovations copy() const;

private:
  Map _nodes, _links;
  ID _nextNodeID, _nextLinkID;

  static ID id (const ID &src, const ID &dst, const Map &map);
  static ID get_id (const ID &src, const ID &dst, Map &map, ID &nextID);
  static ID new_id (const ID &src, const ID &dst, Map &map, ID &nextID);
};

} // end of namespace kgd::eshn::genotype

#endif // KGD_CPPN_GENOTYPE_INNOVATIONS_H
