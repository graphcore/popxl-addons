# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from popart_extensions.tuple_map import TupleMap


def test_tuple_map():
    foo = TupleMap[str, str]()

    foo.insert("foobar", ("a", "b"))
    assert foo.foobar == "b"
    assert foo.tuple_map() == {"a": "b"}


def test_tuple_map_insert_all():
    foo = TupleMap[str, str]()
    foo.insert("foobar", ("a", "b"))
    bar = TupleMap[str, str]()

    bar.insert_all(foo)
    assert bar.foobar == "b"
    assert bar.tuple_map() == {"a": "b"}


def test_tuple_map_insert_child():
    foo = TupleMap[str, str]()
    foo.insert("X", ("a", "b"))
    foo.insert("Y", ("c", "d"))
    bar = TupleMap[str, str]()
    bar.insert("Z", ("e", "f"))
    bar.insert("foo", foo)
    assert bar.foo == foo
    assert bar.foo.X == "b"
    assert bar.tuple_map() == {"a": "b", "c": "d", "e": "f"}


def test_tuple_map_insert_child_with_dunder():
    foo = TupleMap[str, str]()
    foo["X"] = "a", "b"
    foo["Y"] = "c", "d"
    bar = TupleMap[str, str]()
    bar["Z"] = "e", "f"
    bar["foo"] = foo

    assert bar.foo == foo
    assert bar.foo.X == "b"
    assert bar.tuple_map() == {"a": "b", "c": "d", "e": "f"}
