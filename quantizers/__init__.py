from .base import (
    BaseQuantizer,
    NoQ,
)

from .noise import (
    NoiseQuantizer,
)

from .scalar import (
    ScQuestQ,
    ScAbsMaxQ,
)

from .vector import (
    QuipSharpQuestQuantizer,
)

from .trellis import (
    QtipQuestQuantizer,
)

from .higgs import (
    HiggsQuestQuantizer,
)

QUANTIZER_CLASSES = {
    "NoQ": NoQ,
    "NoiseQuantizer": NoiseQuantizer,
    "ScQuestQ": ScQuestQ,
    "ScAbsMaxQ": ScAbsMaxQ,
    "QuipSharpQuestQuantizer": QuipSharpQuestQuantizer,
    "QtipQuestQuantizer": QtipQuestQuantizer,
    "HiggsQuestQuantizer": HiggsQuestQuantizer,
}
