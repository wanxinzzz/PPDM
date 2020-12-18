from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .hoidet import HoidetTrainer
from .embedding import EmbeddingTrainer

train_factory = {
  'hoidet': HoidetTrainer,
  'embedding': EmbeddingTrainer
}
