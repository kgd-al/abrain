from enum import Enum


class NeuronType(Enum):
    INPUT = "Input"
    HIDDEN = "Hidden"
    OUTPUT = "Output"


class Config:
    dimension: int = 3
    max_iterations: int = 10
    activation_function: str = "ssgn"
