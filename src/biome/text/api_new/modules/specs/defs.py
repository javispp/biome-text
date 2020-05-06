import copy
import inspect
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from allennlp.common import FromParams, Params
from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper
from allennlp.modules.seq2vec_encoders import _Seq2VecWrapper
from allennlp.modules.bimpm_matching import BiMpmMatching


def _find_input_attribute(component: Any) -> str:
    """Find the properly input dimension attribute name for a given component"""
    input_dim_attribute = None
    if isinstance(component, (_Seq2SeqWrapper, _Seq2VecWrapper)):
        input_dim_attribute = "input_size"
    elif component is BiMpmMatching:
        input_dim_attribute = "hidden_dim"
    else:
        init_method_keys = inspect.signature(component.__init__).parameters.keys()
        for param_name in ["embedding_dim", "input_dim"]:
            if param_name in init_method_keys:
                input_dim_attribute = param_name
                break
    return input_dim_attribute


T = TypeVar("T")


class ComponentSpec(Generic[T], FromParams):
    """
    The layer spec component allows create Pytorch modules lazily,
    and instantiate them inside a context (Model or other component) dimension layer chain.

    The layer spec wraps a component params and will generate an instance of type T once the input_dim is set.

    """

    @classmethod
    def from_params(cls: Type[T], params: Params, **extras) -> T:
        return cls(**params.as_dict())

    def __resolve_layer_class(self, type_name: Optional[str] = None) -> Type[T]:
        layer_class = getattr(self.__class__, "__orig_bases__")[0].__args__[0]

        return layer_class.by_name(type_name) if type_name else layer_class

    def __init__(self, **config):
        self._config = config or {}
        self._layer_class = self.__resolve_layer_class(self._config.get("type"))

    def input_dim(self, input_dim: int) -> "ComponentSpec":
        """Sets the input dimension attribute for this layer configuration"""
        self.__update_config_with_input_dim(input_dim)
        return self

    def __update_config_with_input_dim(self, input_dim: int):
        input_dim_attribute = _find_input_attribute(self._layer_class)

        if input_dim_attribute:
            self._config[input_dim_attribute] = input_dim

    @property
    def config(self) -> Dict[str, Any]:
        """Component read-only configuration"""
        return copy.deepcopy(self._config)

    def compile(self, **extras) -> T:
        """
        Using the wrapped configuration and the input dimension, generates a
        instance of type T representing the layer configuration
        """
        if not self.config:
            raise ValueError(f"No configuration found for {self}")

        config = self.config
        if "type" in config:
            config.pop("type")

        return self._layer_class.from_params(Params(config), **extras)
