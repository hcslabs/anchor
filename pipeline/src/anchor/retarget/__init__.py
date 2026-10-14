"""Robot embodiment retargeting interfaces and HCS G1 implementation.

The dependency-light G1 solver is shipped here as working HCS code.  Other
retargeting methods may remain closed behind their interface modules.
"""

from .hcs_g1 import G1Retargeter, RetargetSettings, retarget_file
from .embodiment import MorphologyRetargeter, load_embodiment_profile, retarget_embodiment_file

__all__ = [
    "G1Retargeter",
    "RetargetSettings",
    "retarget_file",
    "MorphologyRetargeter",
    "load_embodiment_profile",
    "retarget_embodiment_file",
]
