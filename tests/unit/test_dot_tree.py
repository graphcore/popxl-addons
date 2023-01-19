# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest

from popxl_addons.dot_tree import DotTree


def test_dot_tree_str():
    tree = DotTree.from_dict({"a": 0})
    tree.insert("b", 1)
    tree.insert("c", DotTree(a=2, b=DotTree(a=3)))

    assert tree.a == 0
    assert tree.get("a") == 0
    assert tree.c.b.a == 3
    assert tree.to_dict() == {"a": 0, "b": 1, "c.a": 2, "c.b.a": 3}


def test_dot_tree_int():
    tree = DotTree.from_dict({0: 0})
    tree.insert("1", 1)
    tree.insert(2, DotTree(**{"0": 2, "1": DotTree.from_dict({"0": 3})}))

    assert tree.get(0) == 0
    assert tree.get("0") == 0
    assert tree.get(1) == 1
    assert tree.get("1") == 1
    assert tree.get(2).get(1).get(0) == 3
    assert tree.to_dict() == {"0": 0, "1": 1, "2.0": 2, "2.1.0": 3}


def test_neg_index():
    tree = DotTree()
    tree.insert(0, "a")
    tree.insert(1, "b")
    tree.insert(3, "d")

    assert tree[-1] == "d"
    assert tree[-3] == "b"

    with pytest.raises(KeyError):
        tree[-2]

    with pytest.raises(KeyError):
        tree[-5]


def test_implicit_numeric_conversion():
    tree = DotTree()
    tree.insert(1, "a")

    assert tree[1] == "a"
    assert tree["1"] == "a"
    assert tree.get(1) == "a"
    assert tree.get("1") == "a"
    assert getattr(tree, "1") == "a"

    assert "1" in tree._map
    assert 1 not in tree._map


def test_invalid_keys():
    tree = DotTree()

    with pytest.raises(ValueError):
        tree.insert(-1, "a")

    with pytest.raises(ValueError):
        tree.insert(".a", "a")


def test_nested_clear():
    tree = DotTree(foo=DotTree(bar="baz"))
    assert tree.foo.bar == "baz"
    tree._clear()
    assert tree.foo
    assert not hasattr(tree.foo, "bar")


def test_nested_insert():
    tree = DotTree()
    # Insert
    tree.insert("foo.bar.hee", "a")
    assert tree.foo.bar.hee == "a"

    # Insert with number
    tree.insert("tee.0.hee", "b")
    assert tree.tee[0].hee == "b"

    # Error when key already exists
    with pytest.raises(ValueError):
        tree.insert("foo.bar.goo", "c")

    # Insert when key already exists and merge
    tree.insert("foo.bar.goo", "c", overwrite=True)
    assert tree.foo.bar.goo == "c"
    assert tree.foo.bar.hee == "a"


def test_filter():
    tree = DotTree()
    tree.insert("foo.bar.hee", "a")
    tree.insert("foo.bar.foo", "b", overwrite=True)
    tree.insert("tee.0.hee", "c")

    new_tree = tree.filter_keys(["foo.bar.hee", "tee"])
    assert new_tree.len() == 2
    assert new_tree.foo.bar.hee == "a"
    assert new_tree.tee[0].hee == "c"
