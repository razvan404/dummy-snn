from ..competition import CompetitionMechanism
from ..learning import LearningMechanism
from ..spiking_module import SpikingModule


class SpikingMultilayer(SpikingModule):
    __slots__ = [
        "num_inputs",
        "num_hidden",
        "num_outputs",
        "learning_mechanism",
        "competition_mechanism",
    ]

    def __init__(
        self,
        num_inputs: int,
        num_hidden: list[int],
        num_outputs: int,
        learning_mechanism: LearningMechanism,
        competition_mechanism: CompetitionMechanism | None = None,
    ):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.learning_mechanism = learning_mechanism
        self.competition_mechanism = competition_mechanism
