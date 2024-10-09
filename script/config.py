import abc
import yaml
from dataclasses import dataclass, fields, Field
from lightning import LightningModule, Trainer
from torch.utils.data import Dataset, DataLoader
from typing import *


@dataclass
class Config(abc.ABC):

    def serialize(self) -> Dict[str, Any]:
        def _attr_serialize(field: Field):
            attr = getattr(self, field.name)
            if issubclass(field.type, int):
                return int(attr)
            if issubclass(field.type, bool):
                return bool(attr)
            elif issubclass(field.type, float):
                return float(attr)
            elif issubclass(field.type, str):
                return str(attr)
            elif issubclass(field.type, list):
                return list(attr)
            elif issubclass(field.type, tuple):
                return tuple(attr)
            elif issubclass(field.type, dict):
                return dict(attr)
            elif issubclass(field.type, Config):
                return attr.serialize()

        return {
            field.name: (attr := _attr_serialize(field))
            for field in fields(self.__class__)
            if attr is not None
        }

    @classmethod
    def deserialize(cls, config_dict: Dict[str, Any]) -> "Config":
        def _attr_deserialize(field: Field):
            attr = config_dict[field.name]
            if issubclass(field.type, int):
                return int(attr)
            if issubclass(field.type, bool):
                return bool(attr)
            elif issubclass(field.type, float):
                return float(attr)
            elif issubclass(field.type, str):
                return str(attr)
            elif issubclass(field.type, list):
                return list(attr)
            elif issubclass(field.type, tuple):
                return tuple(attr)
            elif issubclass(field.type, dict):
                return dict(attr)
            elif issubclass(field.type, Config):
                return field.type.deserialize(attr)

        kwargs = {
            field.name: _attr_deserialize(field)
            for field in fields(cls)
            if field.name in config_dict
        }
        return cls(**kwargs)
