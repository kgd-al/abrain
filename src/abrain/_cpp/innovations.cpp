#include <stdexcept>
#include <iostream>

#include "innovations.h"

#ifndef NDEBUG
//#define DEBUG_INNOVATIONS 1
#endif

namespace kgd::eshn::genotype {
using ID = Innovations::ID;

Innovations::Innovations() : _nextNodeID(0), _nextLinkID(0) {}
Innovations::Innovations(const Map &&nodes, const Map &&links,
                         const ID &nextNodeId, const ID &nextLinkID)
                         : _nodes(nodes), _links(links),
                         _nextNodeID(nextNodeId), _nextLinkID(nextLinkID) {}

void Innovations::initialize(ID nextNodeID) {
  _nextNodeID = nextNodeID;
  _nextLinkID = 0;
  _nodes.clear();
  _links.clear();
}

ID Innovations::link_id(const ID &src, const ID &dst) const {
  return id(src, dst, _links);
}

ID Innovations::node_id(const ID &src, const ID &dst) const {
  return id(src, dst, _nodes);
}

ID Innovations::id(const ID &src, const ID &dst, const Map &map) {
  auto it = map.find(std::make_pair(src, dst));
  if (it != map.end())
    return it->second;
  else
    return NOT_FOUND;
}

ID Innovations::get_link_id(const ID &src, const ID &dst) {
#if DEBUG_INNOVATIONS
//  std::cerr << "[innov-debug] add_link_innovation(" << src << " -> " << dst
//            << ")" << std::endl;
#endif
  return get_id(src, dst, _links, _nextLinkID);
}

ID Innovations::get_node_id(const ID &src, const ID &dst) {
#if DEBUG_INNOVATIONS
  std::cerr << "[innov-debug] add_node_innovation(" << src << " -> " << dst
            << ")" << std::endl;
#endif
  return get_id(src, dst, _nodes, _nextNodeID);
}

ID Innovations::get_id(const ID &src, const ID &dst, Map &map, ID &nextID) {
  auto key = std::make_pair(src, dst);
  auto it = map.find(key);
  if (it == map.end()) {
#if DEBUG_INNOVATIONS
    std::cerr << "[innov-debug] >>>> Not found. Inserting "
              << map.size() << "th item" << std::endl;
#endif
    auto id = nextID++;
    map[key] = id;
    return id;
  } else {
#if DEBUG_INNOVATIONS
    std::cerr << "[innov-debug] >>>> Found. Returning " << it->second << std::endl;
#endif
    return it->second;
  }
}


ID Innovations::new_link_id(const ID &src, const ID &dst) {
#if DEBUG_INNOVATIONS
  std::cerr << "[innov-debug] add_link_innovation(" << src << " -> " << dst
            << ")" << std::endl;
#endif
  return new_id(src, dst, _links, _nextLinkID);
}

ID Innovations::new_node_id(const ID &src, const ID &dst) {
#if DEBUG_INNOVATIONS
  std::cerr << "[innov-debug] add_node_innovation(" << src << " -> " << dst
            << ")" << std::endl;
#endif
  return new_id(src, dst, _nodes, _nextNodeID);
}

ID Innovations::new_id(const ID &src, const ID &dst, Map &map, ID &nextID) {
  auto key = std::make_pair(src, dst);
  auto id = nextID++;
#if DEBUG_INNOVATIONS
  auto it = map.find(key);
  std::cerr << "[innov-debug] >>>> Force inserting at " << id;
  if (it == map.end())
    std::cerr << " instead of " << it->second;
  std::cerr << std::endl;
#endif
  map[key] = id;
  return id;
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
