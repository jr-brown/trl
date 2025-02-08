"""A dummy script with rough code for PER implementation.
In particular trying miscellaneous things to take advantage of language server abilities."""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List
from torch import Tensor


@dataclass
class TensorSamples(ABC):
    """Holding collections of tensors of shared 'batch' dimension."""

    def __init__(self, tensor_names: List[str]):
        assert len(tensor_names) > 0, "At least one tensor name must be provided."
        self.tensor_names = tensor_names
        self._check_consistent_batch_dimension()

    @property
    def _batch_size(self):
        """Return the size of the batch dimension."""
        return getattr(self, self.tensor_names[0]).shape[0]

    @property
    def _tensor_dict(self):
        """Return a dictionary mapping tensor names to tensors."""
        return {name: getattr(self, name) for name in self.tensor_names}

    def _check_consistent_batch_dimension(self):
        """Check that all tensors have the same batch dimension."""
        assert all(
            getattr(self, name).shape[0] == self._batch_size
            for name in self.tensor_names
        ), "All tensors must have the same (leading) batch dimension."

    def subset(self, indices: List[int]):
        """Return a new instance with only the specified indices."""
        return self.__class__(
            **{name: getattr(self, name)[indices] for name in self.tensor_names}
        )

    # TODO: Check that this interacts as desired with a particular instance.


@dataclass
class BufferTensors(TensorSamples):
    """Collection of tensors required for PER i.o."""

    query_responses: Tensor
    ref_logprobs: Tensor
    gen_logprobs: Tensor
    returns: Tensor
    priorities: Tensor

    def __post_init__(self):
        super().__init__(
            ["query_responses", "ref_logprobs", "gen_logprobs", "returns", "priorities"]
        )
