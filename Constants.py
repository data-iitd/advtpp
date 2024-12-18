from enum import Enum


KL_MAX = 10000


class EnumMixin(object):
    def __str__(self):
        return self.name

    def to_json(self):
        return self.__str__()

    @classmethod
    def from_string(cls, s):
        try:
            return cls[s]
        except KeyError:
            raise ValueError()


class ArchitectureChoice(EnumMixin, Enum):
    THP = 0
    RNN_TPP = 1


class AttackRegularizerChoice(EnumMixin, Enum):
    """
    The kind of regularizer to add to the loss when using
    our attack.
    """
    NONE = 0
    KLDIV = 1
    KLDIV_BETA = 2
    HELLINGER = 3


class BlackBoxSubtype(EnumMixin, Enum):
    """
    Black box variations decided according to Florian Tramer's Ensemble Attacks
    paper. See its appendix for formal definitions of threat models.
    These options have an effect only in the part of the code where the training set is chosen.
    Other than that, these are only meant to be a threat model subtype identifier.
    """
    RANDOMNESS = 0 # tgt and src differ in random init
    ARCH = 1 # tgt and src differ in arch: number of params or a different arch altogether
    TRAINSET = 2 # tgt trained on full trainset; src train on a randomly sampled subset of the trainset


class CleanTrainsetChoice(EnumMixin, Enum):
    """
    When training a clean model, which model is it?
    """
    TARGET = 0
    SOURCE = 1


class NoiseActivationFunction(EnumMixin, Enum):
    RELU = 0
    LEAKY_RELU = 1
    TANH = 2


class NoiseModelChoice(EnumMixin, Enum):
    # If noise_uniform specified, just sample from the uniform dist.
    # in the range specified by min. inter-event time.
    UNIFORM_NOISE = 0
    NOISE_RNN = 1 # unused
    NOISE_MLP = 2 # unused
    NOISE_TRANSFORMER = 3
    NOISE_TRANSFORMER_V2 = 4
    NOISE_SPARSE_NORMAL = 5


class TrainMode(EnumMixin, Enum):
    """
    Training mode to choose from when using our attack.
    """
    ADV_LLH = 0
    ADV_LLH_DIAG = 1


class TrainTimeAttack(EnumMixin, Enum):
    OUR = 0
    TRADES = 1 # unused
    MART = 2 # unused
    NONE = 3
    FGSM = 4
    PGD = 5
    AUTO_ATTACK = 6
    MI_FGSM = 7
    VANILLA_NOISE = 8
    OUR_PLUS_DEFENSE = 9
    TS_DET = 10
    TS_PROB = 11
    # XXX: add time series attacks


class OperationMode(EnumMixin, Enum):
    # Attack a CLEAN defender model, write stats and save checkpoints.
    ATTACK_EVAL = 0
    # Attack a CLEAN defender model and train the defender on adv examples so generated.
    # Then evaluate TRAINED defender according to other methods. For example, if defender trained on OUR,
    # then evaluate by all other methods available.
    ATTACK_PLUS_DEFENSE = 1


class ThreatModel(EnumMixin, Enum):
    WHITE_BOX = 0
    BLACK_BOX = 1
    WHITE_BOX_SOURCE = 2