from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from pathlib import Path
from typing import Literal

from data_manipulation.lung_caption_vocab import Vocabulary


class LungRNN(nn.Module, metaclass=ABCMeta):

    @property
    @abstractmethod
    def vocab(self) -> Vocabulary:
        pass

    @abstractmethod
    def caption(self,
                features: torch.Tensor,
                max_length: int) -> str:
        pass

    @abstractmethod
    def caption_raw_tokens(self,
                           features: torch.Tensor,
                           max_length: int) -> list:
        pass

    @abstractmethod
    def forward(self,
                features: torch.Tensor,
                max_length: int) -> torch.Tensor:
        pass

    @abstractmethod
    def save_model_and_vocab(self,
                             model_dir: str | Path,
                             vocab_dir: str | Path):
        pass
