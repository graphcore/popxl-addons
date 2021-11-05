# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import re
from typing import Generic, Tuple, TypeVar, Union, Dict, ItemsView

A = TypeVar("A")
B = TypeVar("B")

# for `to_dict` method
TupleMapDict = Dict[str, Union[Tuple[A, B], 'TupleMapDict[A, B]']]


def sanitise(name: str) -> str:
    return re.sub(r'\W|^(?=\d)', '_', name)


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
    ```
    """
    def __init__(self):
        self._map: Dict[str, Union[Tuple[A, B], 'TupleMap[A, B]']] = {}

    def insert_all(self, other: 'TupleMap[A, B]'):
        """
        Insert all items from `other` into self.
        Merge duplicate keys that have the same tuple, otherwise throw an error.
        """
        keys_intersection = set(self._map.keys()).intersection(other._map.keys())
        other_ = other.copy()
        for key in keys_intersection:
            if self._map[key] != other._map[key]:
                if isinstance(self._map[key], TupleMap) and isinstance(other._map[key], TupleMap):
                    self._map[key].insert_all(other._map[key])
                    del other_._map[key]
                else:
                    raise ValueError(
                        f"Duplicate key found but with two different values that cannot be merged. Key: {key}. "
                        f"Self value: {self._map[key]}. Other Value {other._map[key]}")
        self._map.update(other_._map)

    def __getattr__(self, key: str):
        """Returns 2nd value of tuple or child TupleMap"""
        _map = self.__getattribute__("_map")
        if key in _map and isinstance(_map[key], tuple):
            return _map[key][1]
        if key in _map and isinstance(_map[key], TupleMap):
            return _map[key]
        return self.__getattribute__(key)

    def __getitem__(self, key: str):
        """Get item with key"""
        return self._map[key]

    def __setitem__(self, key: str, value: Union[Tuple[A, B], 'TupleMap[A, B]']):
        """Set item with key. Either a tuple or child TupleMap"""
        self._validate_key(key)
        self._map[key] = value

    insert = __setitem__

    def __len__(self):
        return len(self._map)

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
        mapping = {}
        for value in self._map.values():
            if isinstance(value, tuple):
                mapping[value[0]] = value[1]
            else:
                # Check for name conflicts
                nested_tp = value.tuple_map()
                duplicate_keys = set(mapping.keys()).intersection(nested_tp.keys())
                if len(duplicate_keys) > 0:
                    raise ValueError(f"Duplicate keys exist in nested tuplemap: {', '.join(duplicate_keys)}")
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
        except:  # Fail gracefully and always provide a string
            return super().__str__()

    def to_dict(self) -> TupleMapDict[A, B]:
        return {k: (v if isinstance(v, tuple) else v.to_dict()) for k, v in self._map.items()}

    def _validate_key(self, key: str):
        if key in self._map:
            raise ValueError(f"'{key}' already exists in {self.__repr__()}")
        if not key.isidentifier():
            raise ValueError(f"'{key}' is not a valid python identifier")

    def copy(self):
        """Return shallow copy"""
        other = TupleMap()
        other._map = self._map.copy()
        return other
