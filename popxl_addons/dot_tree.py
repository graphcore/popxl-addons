# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import re
from typing import Callable, Dict, Generic, Iterable, List, Tuple, Type, TypeVar, Union, Mapping
import typing_extensions

from popxl_addons.utils import OrderedDict


def sanitise(name: str) -> str:
    """Remove characters from string that prevent it from being a python identifier"""
    return re.sub(r'\W|^(?=\d)', '_', name)


L = TypeVar("L")
K = TypeVar("K")
V = TypeVar("V")
CLS = TypeVar("CLS", bound='DotTree')


def to_mapping(key: 'DotTree[K]', value: 'DotTree[V]') -> Dict[K, V]:
    """Given two DotTrees, for common keys create a dictionary of their values"""
    key_dict = key.to_dict()
    value_dict = value.to_dict()

    key_keys = set(value_dict.keys())
    value_keys = set(key_dict.keys())

    mapping = {}
    for common in key_keys.intersection(value_keys):
        mapping[key_dict[common]] = value_dict[common]

    return mapping


class DotTree(Generic[L]):
    """Generic tree container.
        Children or leaf nodes can be accessed using dot notation:

        .. code-block:: python
            tree = DotTree[str](foo="bar")
            tree.foo == "bar"

            tree = ExampleTree.from_dict({"foo.bar": 1})
            tree.foo.bar == 1

        Numeric keys can be used and accessed with `[]`:

        .. code-block:: python
            tree = DotTree[str]()
            tree.insert(1, "baz")
            tree[1] == "baz"

        Serialisation to/from dictionaries is available with `to_dict`/`from_dict`. 
        As well as to/from lists with `unpack`/`pack`.
    """

    def __init__(self: CLS, **kwargs: 'Union[L, CLS]'):
        self._map: OrderedDict[str, Union[L, CLS]] = OrderedDict()
        for k, v in kwargs.items():
            self.insert(k, v)

    def __getattr__(self, key: str):
        try:
            return self.get(key)
        except KeyError as ke:
            pass
        try:
            return self.__getattribute__(key)
        except AttributeError as ae:
            keys = "    \n".join(self._map.keys())
            raise AttributeError(f"No attribute '{key}'. Available Keys:\n{keys}") from ae

    def __getitem__(self: CLS, key: Union[str, int]) -> 'Union[L, CLS]':
        return self.get(key)

    def get(self: CLS, key: Union[str, int]) -> 'Union[L, CLS]':
        """Get a value. Ints and strings that are numerical are interpreted as a numerical key."""
        existing_keys = "    \n".join(self._map.keys())

        # Int key
        if isinstance(key, int) or key.isnumeric():
            key = int(key)
            if key < 0:
                numeric_max = max(map(int, filter(lambda e: e.isnumeric(), self._map.keys())))
                pos_key = numeric_max + 1 + key
                if pos_key < 0:
                    raise KeyError(f"Negative numerical key does not exist: {key}. Max numerical key: {numeric_max}. "
                                   f"Available keys: {existing_keys}")
                if str(pos_key) not in self._map:
                    raise KeyError(f"Negative numerical key does not exist: {key}. Equivalent positive key: {pos_key}. "
                                   f"Available keys: {existing_keys}")
                key = pos_key
            else:
                if str(key) not in self._map:
                    raise KeyError(f"Numerical key does not exist: {key}. Available keys: {existing_keys}")
            return self._map[str(key)]

        key, *other_keys = key.split('.')

        if key in self._map:
            # Return singleton
            if len(other_keys) == 0:
                return self._map[key]

            # Return sub-keys
            if isinstance(self._map[key], type(self)):
                return self._map[key].get('.'.join(other_keys))
            else:
                raise KeyError(f"Key exists ({key}) but is not a sub-DotTree of type {type(self)} "
                               f"and therefore you cannot access the sub-key: {other_keys}.")

        raise KeyError(f"Key does not exist: {key}. Available keys: {existing_keys}")

    def insert(self: CLS, key: Union[str, int], value: Union[L, CLS], overwrite: bool = False):
        """Set item with key. Numerical keys represented using an int will automatically be converted to a string.
        If the key contains a dot this will be interpreted as a nested DotTree.
        If overwrite==True, nested dot trees will be merged and values will be overwritten."""
        if isinstance(key, int):
            if key < 0:
                raise ValueError(f"Numerical key cannot be negative: {key}")
            key = str(key)
        key, *other_keys = key.split('.')
        self._validate_key(key, overwrite)
        if len(other_keys) == 0:
            self._map[key] = value
        else:
            if key in self._map and isinstance(self._map[key], type(self)):
                # Copy so if nested insert fails we can safely roll back
                nested_dottree = self._map[key].copy()
            else:
                # If key is V then override
                nested_dottree = type(self)()

            nested_dottree.insert('.'.join(other_keys), value, overwrite)
            # Only insert if nested operations don't fail
            self._map[key] = nested_dottree

    def update(self: CLS, values: CLS, overwrite: bool = False):
        """Update map with another map"""
        for k, v in values.copy()._map.items():
            self.insert(k, v, overwrite)

    def pop(self, key: str):
        """Pop key."""
        key, *other_keys = key.split('.')
        if len(other_keys):
            return self.get(key).pop('.'.join(other_keys))
        else:
            return self._map.pop(key)

    def copy(self):
        """Return shallow copy"""
        tree = self.__class__()
        for k, v in self._map.items():
            if isinstance(v, DotTree):
                v = v.copy()
            tree.insert(k, v)
        return tree

    def filter_keys(self, keys: Iterable[Union[str, int]]) -> 'CLS':
        """Return a new DotTree that only include the given keys."""
        keys = list(keys)
        if len(keys) != len(set(keys)):
            raise ValueError("`keys` cannot contain duplicate values.")
        new_dottree = self.__class__()
        for key in keys:
            value = self.get(key)
            new_dottree.insert(key, value, overwrite=True)

        return new_dottree

    def keys(self) -> List[str]:
        return list(self._map.keys())

    def keys_flat(self) -> List[str]:
        return list(self.to_dict().keys())

    def values(self) -> List[V]:
        return list(self._map.values())

    def values_flat(self) -> List[V]:
        return list(self.to_dict().values())

    def len(self) -> int:
        return len(self._map)

    def len_flat(self) -> int:
        return len(self.to_dict())

    def to_mapping(self, values: 'DotTree[V]') -> Dict[L, V]:
        """Given another DotTree, for common keys create a dictionary of their values"""
        return to_mapping(self, values)

    def to_dict(self) -> 'OrderedDict[str, L]':
        """Output a dict of the DotTree. Keys with dots represent nested DotTrees"""
        mapping = OrderedDict()
        for key, value in self._map.items():
            if isinstance(value, DotTree):
                mapping.update({f"{key}.{name}": v for name, v in value.to_dict().items()})
            else:
                mapping[key] = value
        return mapping

    @classmethod
    def from_dict(cls: Type[CLS], d: Mapping[Union[str, int], Union[L, 'DotTree[L]']]) -> 'CLS':
        """
        Create a DotTree from a dictionary.
        Keys in the dictionary that contain dots will create a nested DotTree structure.
        """
        tree = cls()
        for flatkey, value in d.items():
            if isinstance(flatkey, int):
                flatkey = str(flatkey)
            keys = flatkey.split(".")
            self = tree
            while len(keys) > 1:
                key = keys.pop(0)
                if not hasattr(self, key):
                    self.insert(key, cls())
                self = getattr(self, key)
            self.insert(keys.pop(0), value)
        return tree

    def unpack(self) -> Tuple[List[str], List[L]]:
        """Return a list of keys and a list of values sorted by key."""
        name_leaf = self.to_dict()
        if name_leaf:
            return zip(*sorted(name_leaf.items(), key=lambda l: l[0]))
        return [], []

    @classmethod
    def pack(cls, keys: Iterable[Union[str, int]], values: Iterable[L]):
        """
        Create a DotTree from a list of keys and list of values.
        Keys in the dictionary that contain dots will create a nested DotTree structure.
        """
        return cls.from_dict(dict(zip(keys, values)))

    def map(self: CLS, fn: Callable[[L], L]) -> CLS:
        """
        Apply fn to each element of the collection and returns a new DotTree with the transformed values.
        """
        mapped = OrderedDict()
        for k, v in self.to_dict().items():
            mapped[k] = fn(v)
        return self.from_dict(mapped)

    def _clear(self):
        """Empty all entries including all nested DotTrees"""
        for k, v in list(self._map.items()):
            if isinstance(v, DotTree):
                v._clear()
            else:
                self._map.pop(k)

    def _validate_key(self, key: str, allow_mutable: bool = False):
        if not allow_mutable and key in self._map:
            raise ValueError(f"'{key}' already exists")
        if not key.isnumeric() and not key.isidentifier():
            raise ValueError(f"'{key}' is not a valid python identifier")

    def __repr__(self) -> str:
        try:
            cls_name = type(self).__name__
            return f'{cls_name}({self.to_dict()})'
        except:  # Fail gracefully and always provide a string
            return super().__repr__()

    def __contains__(self, key):
        try:
            self.get(key)
            return True
        except KeyError:
            return False
