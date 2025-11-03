"""
obj_config.py
=============

Lightweight attribute-accessible dictionary with optional immutability.

This module defines a single class:

- `AttrDict`: a `dict` subclass that lets you access keys as attributes
  (e.g., `cfg.learning_rate` instead of `cfg['learning_rate']`), and supports
  toggling immutability recursively so that you can freeze a configuration
  object after construction to prevent accidental modifications.

Key behaviors
-------------
- Attribute access mirrors dictionary access:
    * `__getattr__` checks `__dict__` first (normal Python attributes), then the
      mapping (`self[name]`). If missing, it raises `AttributeError`.
    * `__setattr__` writes to `__dict__` if the attribute exists there; otherwise
      it writes into the mapping (`self[name] = value`).
- Immutability:
    * When `immutable(True)` is called, any subsequent attribute assignment
      or item assignment through `__setattr__` will raise `AttributeError`.
    * The immutable flag is stored under a reserved attribute name
      `AttrDict.IMMUTABLE` inside `__dict__` and is **not** part of the key space.
    * The `immutable()` method applies the flag recursively to any nested
      `AttrDict` instances found both in attributes (`__dict__`) and in values
      of the mapping itself (`self.values()`).
"""


class AttrDict(dict):
    IMMUTABLE = "__immutable__"     # reserved key used internally in __dict__

    def __init__(self, *args, **kwargs):
        # Initialize the underlying dict normally first
        super(AttrDict, self).__init__(*args, **kwargs)
        # Initialize the immutability flag in __dict__. Using __dict__ directly
        # avoids going through __setattr__ (which enforces immutability).
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        """
        Attribute-style read access.

        Resolution order:
        1) If `name` exists as a *real* attribute in `__dict__`, return it.
        2) Else, if `name` is a key in the mapping, return `self[name]`.
        3) Else, raise AttributeError to follow normal Python semantics.
        """
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            # Important: raise AttributeError (not KeyError) so tools like
            # hasattr()/getattr() work as expected with attribute access.
            raise AttributeError(name)

    def __setattr__(self, name, value):
        """
        Attribute-style write access.

        Behavior:
        - If the object is immutable, *any* attempt to set attributes or keys
          will raise an AttributeError.
        - Otherwise:
            * If `name` already exists in `__dict__`, write to `__dict__`.
            * Else, write to the mapping (`self[name] = value`).

        Note:
        - This keeps a clean separation between actual object attributes
          (stored in `__dict__`, e.g., the immutability flag) and user-provided
          configuration fields (stored as mapping entries).
        """
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                # Update a real attribute (e.g., internal flags/utilities)
                self.__dict__[name] = value
            else:
                # Create/update a config field in the mapping
                self[name] = value
        else:
            # When immutable, disallow *all* writes through attribute access
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.format(
                    name, value
                )
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.

        Parameters
        ----------
        is_immutable : bool
            True to freeze the object (and nested AttrDicts), False to unfreeze.

        Details
        -------
        - This method flips the internal flag stored in `__dict__` under the
          reserved key `AttrDict.IMMUTABLE`.
        - Recursion runs over:
            * values in `__dict__` (object attributes), and
            * values of the mapping (`self.values()`).
          If any of those values are themselves `AttrDict` instances, the same
          immutability state is applied to them.
        """
        # Toggle the local immutability flag
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable

        # Recursively set immutable state on nested AttrDicts found as attributes
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        
        # Recursively set immutable state on nested AttrDicts found as dict values
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        """
        Return the current immutability state.

        Returns
        -------
        bool
            True if the object is currently frozen (immutable), else False.
        """
        return self.__dict__[AttrDict.IMMUTABLE]
