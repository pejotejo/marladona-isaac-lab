"""Soccer observation, reward, termination, and curriculum terms."""

from mjlab.envs.mdp.terminations import time_out as time_out

from . import observations as observations
from . import rewards as rewards
from . import terminations as terminations
from .curriculums import field_levels as field_levels
