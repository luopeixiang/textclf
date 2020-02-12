"""
本文件的代码大部分参考自 pytext:
https://github.com/facebookresearch/pytext/blob/master/pytext/config/pytext_config.py
"""


from collections import OrderedDict
import json


class ConfigBaseMeta(type):
    def annotations_and_defaults(cls):
        annotations = OrderedDict()
        defaults = {}
        for base in reversed(cls.__bases__):  # 从基类开始更新
            if base is ConfigBase:
                continue
            annotations.update(getattr(base, "__annotations__", {}))
            defaults.update(getattr(base, "_field_defaults", {}))
        annotations.update(vars(cls).get("__annotations__", {}))
        defaults.update({k: getattr(cls, k) for k in annotations if hasattr(cls, k)})
        return annotations, defaults

    @property
    def __annotations__(cls):
        annotations, _ = cls.annotations_and_defaults()
        return annotations

    _field_types = __annotations__

    @property
    def _fields(cls):
        return cls.__annotations__.keys()

    @property
    def _field_defaults(cls):
        _, defaults = cls.annotations_and_defaults()
        return defaults


class ConfigBase(metaclass=ConfigBaseMeta):
    def __init__(self, config_dict={}):
        """Configs can be constructed by specifying values by keyword.
        If a keyword is supplied that isn't in the config, or if a config requires
        a value that isn't specified and doesn't have a default, a TypeError will be
        raised."""
        specified = config_dict.keys() | type(self)._field_defaults.keys()
        required = type(self).__annotations__.keys()
        # Unspecified fields have no default and weren't provided by the caller
        unspecified_fields = required - specified
        if unspecified_fields:
            raise TypeError(f"Failed to specify {unspecified_fields} for {type(self)}")

        # Overspecified fields are fields that were provided but that the config
        # doesn't know what to do with, ie. was never specified anywhere.
        overspecified_fields = specified - required
        if overspecified_fields:
            raise TypeError(
                f"Specified non-existent fields {overspecified_fields} for {type(self)}"
            )

        vars(self).update(config_dict)

    def items(self):
        return self.asdict().items()

    def asdict(self):
        return {k: getattr(self, k) for k in type(self).__annotations__}

    def _replace(self, **kwargs):
        args = self.asdict()
        args.update(kwargs)
        return type(self)(**args)

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in sorted(self.asdict().items()):
            lines += f"{key}: {val}".split("\n")
        return "\n    ".join(lines)

    def __eq__(self, other):
        """Mainly a convenience utility for unit testing."""
        return type(self) == type(other) and self.asdict() == other.asdict()

    def asdict_deep(self):
        res = {}
        field_defaults = type(self)._field_defaults
        for k in field_defaults:
            obj = field_defaults[k]
            if isinstance(obj, ConfigBase):
                res[k] = {}
                res[k]['__class__'] = obj.__class__.__name__
                res[k]['params'] = obj.asdict_deep()
            else:
                res[k] = obj
        return res

    def dump(self, filename):
        """dump default config to file"""
        json_data = {}
        json_data['__class__'] = self.__class__.__name__
        json_data['params'] = self.asdict_deep()
        with open(filename, 'w') as fd:
            json.dump(json_data, fd, indent=4)
