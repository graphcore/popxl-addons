# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# pyright: strict
from typing import Generic, Tuple, TypeVar, Union, Dict, ItemsView

A = TypeVar("A")
B = TypeVar("B")

# for `to_dict` method
TupleMapDict = Dict[str, Union[Tuple[A, B], 'TupleMapDict[A, B]']]


class TupleMap(Generic[A, B]):
    """
    A dictionary where values are either a tuple (length 2) or a TupleMap.
    Key must be strings and a valid python identifier so it can be accessed as a object attribute.
    The TupleMap is key-wise immutable. Nesting of TupleMaps provides a tree structure.

    For example a TupleMap with 3 keys, two of the values are tuples and 1 is a nested `TupleMap`.
    The nested TupleMap has two keys with tuple values
    ```
    TupleMap
    ↳ A → (1, 2)
    ↳ B → (3, 4)
    ↳ C → D → (5, 6)
        ↳ E → (7, 8)
    ```

    You can access elements via:
        - `tm[a]` - using dict notation will return a tuple or child TupleMap
        - `tm.a` - using dot notation will return the last item in the tuple or child TupleMap

    Example:
    ```
    foo = TupleMap[int, int]()
    foo['a'] = 1, 2
    foo['b'] = 3, 4
    bar = TupleMap[int, int]()
    bar['c'] = 5, 6
    foo['bar'] = bar

    assert foo['a'] == (1, 2)
    assert foo.b == 4
    assert foo.tuple_map() == {1: 2, 3: 4, 5: 6}
    """

    def __init__(self):
        self._map: Dict[str, Union[Tuple[A, B], 'TupleMap[A, B]']] = {}

    def insert_all(self, other: 'TupleMap[A, B]'):
        """Insert all items from `other` into self"""
        keys_intersection = set(self._map.keys()).intersection(other._map.keys())
        if len(keys_intersection) > 0:
            raise ValueError(
                f"{self.__repr__()} already contains keys found in {other}: {', '.join(keys_intersection)}")
        self._map.update(other._map)

    def __getattr__(self, key: str):
        """Returns 2nd value of tuple or child TupleMap"""
        if key == "_map":
            return self.__getattribute__(key)
        if key in self._map and isinstance(self._map[key], tuple):
            return self._map[key][1]
        if key in self._map and isinstance(self._map[key], TupleMap):
            return self._map[key]
        return self.__getattribute__(key)

    def __getitem__(self, key: str):
        """Get item with key"""
        return self._map[key]

    def __setitem__(self, key: str, value: Union[Tuple[A, B], 'TupleMap[A, B]']):
        """Set item with key. Either a tuple or child TupleMap"""
        self._validate_key(key)
        self._map[key] = value

    insert = __setitem__

    def tuple_map(self) -> Dict[A, B]:
        """
        Creates a dictionary using the tuple as keys and values.
        Recursively inserts tuples from nested TupleMaps.

        Example:
        ```
        foo = TupleMap[str, str]()
        bar = TupleMap[str, str]()
        foo["X"] = "a", "b"
        foo["Y"] = "c", "d"
        bar["Z"] = "e", "f"
        bar["foo"] = foo
        assert bar.tuple_map() == {"a": "b", "c": "d", "e": "f"}
        ```
        """
        # TODO: should we check for name conflicts?
        mapping = {}
        for value in self._map.values():
            if isinstance(value, tuple):
                mapping[value[0]] = value[1]
            else:
                mapping.update(value.tuple_map())
        return mapping

    def a_map(self) -> Dict[str, A]:
        """Return dict that only includes first item of tuple. Skip child nodes."""
        return {k: v[0] for k, v in self._map.items() if isinstance(v, tuple)}

    def b_map(self) -> Dict[str, B]:
        """Return dict that only includes second item of tuple. Skip child nodes."""
        return {k: v[1] for k, v in self._map.items() if isinstance(v, tuple)}

    def items(self) -> ItemsView[str, Union[Tuple[A, B], 'TupleMap[A, B]']]:
        return self._map.items()

    def __str__(self) -> str:
        try:
            return str(self.to_dict())
        except:
            return super().__str__()

    def to_dict(self) -> TupleMapDict[A, B]:
        return {k: (v if isinstance(v, tuple) else v.to_dict()) for k, v in self._map.items()}

    def _validate_key(self, key):
        if key in self._map:
            raise ValueError(f"'{key}' already exists in {self.__repr__()}")
        if not key.isidentifier():
            raise ValueError(f"'{key}' is not a valid python identifier")
