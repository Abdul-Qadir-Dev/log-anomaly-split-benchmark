from dataclasses import dataclass


@dataclass
class ArmContext:
    dataset: str
    selftest: bool
    n_vocab: int
    seed: int = 0
