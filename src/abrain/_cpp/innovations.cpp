#include <stdexcept>
#include <iostream>

#include "innovations.h"

namespace kgd::eshn::genotype {
using ID = Innovations::ID;

Innovations::Innovations() : _nextNodeID(0), _nextLinkID(0) {}
Innovations::Innovations(const Map &&nodes, const Map &&links,
                         const ID &nextNodeId, const ID &nextLinkID)
                         : _nodes(nodes), _links(links),
                         _nextNodeID(nextNodeId), _nextLinkID(nextLinkID) {}

void Innovations::initialize(ID nextNodeID) {
  _nextNodeID = std::max(_nextNodeID, nextNodeID);
}

ID Innovations::link_innovation_id(const ID &from, const ID &to) {
  return innovation_id(from, to, _links);
}

ID Innovations::node_innovation_id(const ID &from, const ID &to) {
  return innovation_id(from, to, _nodes);
}

ID Innovations::innovation_id(const ID &from, const ID &to, const Map &map) {
  auto it = map.find(std::make_pair(from, to));
  if (it != map.end())
    return it->second;
  else
    return NOT_FOUND;
}

ID Innovations::add_link_innovation(const ID &from, const ID &to) {
  std::cerr << "[kgd-debug] add_link_innovation(" << from << " -> " << to
            << ")" << std::endl;
  return add_innovation(from, to, _links, _nextLinkID);
}

ID Innovations::add_node_innovation(const ID &from, const ID &to) {
  std::cerr << "[kgd-debug] add_node_innovation(" << from << " -> " << to
            << ")" << std::endl;
  return add_innovation(from, to, _nodes, _nextNodeID);
}

ID Innovations::add_innovation(const ID &from, const ID &to,
                               Map &map, ID &nextID) {
  auto key = std::make_pair(from, to);
  auto it = map.find(key);
  std::cerr << "[kgd-debug] >> Trying to insert <" << from << ", " << to << ">" << std::endl;
  if (it == map.end()) {
    auto tmp_size_before = map.size();
    std::cerr << "[kgd-debug] >>>> Not found. Inserting " << tmp_size_before << "th item" << std::endl;
    auto id = nextID++;
    map[key] = id;
    std::cerr << "[kgd-debug] >>>> new size: " << map.size() << std::endl;
    return id;
  } else {
    std::cerr << "[kgd-debug] >>>> Found. Returning " << it->second << std::endl;
    return it->second;
  }
}

Innovations Innovations::copy() const {
  Innovations i;
  i._nodes = _nodes;
  i._links = _links;
  i._nextNodeID = _nextNodeID;
  i._nextLinkID = _nextLinkID;
  return i;
}

} // end of namespace kgd::eshn::genotype
