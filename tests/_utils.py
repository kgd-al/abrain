from pyne.core.genome import Genome


def assert_equal(lhs: Genome, rhs: Genome):
    assert lhs is not rhs
    assert lhs.nodes is not rhs.nodes
    assert lhs.links is not rhs.nodes

    assert len(lhs.nodes) == len(rhs.nodes)
    assert len(lhs.links) == len(rhs.links)

    for lhs_n, rhs_n in zip(lhs.nodes, rhs.nodes):
        assert lhs_n is not rhs_n
        assert lhs_n.id == rhs_n.id
        assert lhs_n.func == rhs_n.func

    for lhs_l, rhs_l in zip(lhs.links, rhs.links):
        assert lhs_l is not rhs_l
        assert lhs_l.id == rhs_l.id
        assert lhs_l.src == rhs_l.src
        assert lhs_l.dst == rhs_l.dst
        assert lhs_l.weight == rhs_l.weight

    assert lhs.nextNodeID == rhs.nextNodeID
    assert lhs.nextLinkID == rhs.nextLinkID
