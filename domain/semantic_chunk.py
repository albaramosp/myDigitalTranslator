from dataclasses import dataclass
import numpy as np

@dataclass
class SemanticChunk:
    text: str
    embedding: np.ndarray