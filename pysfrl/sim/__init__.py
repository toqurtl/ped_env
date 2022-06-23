"""Numpy implementation of the Social Force model."""

__version__ = "1.1.2"

from .simulator import Simulator
from .force.potentials import PedPedPotential, PedSpacePotential
from .force.forces import *
from .utils import plot
