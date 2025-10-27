from .base import BKTModelBase


class BKTModel(BKTModelBase):
    """
    Bayesian Knowledge Tracing (BKT) model implementation.

    This class implements the standard BKT model using Bayesian inference.
    It extends the BKTModelBase class and provides methods for fitting the model,
    predicting student knowledge states, and generating additional quantities.

    Parameters
    ----------
    stan_file : str or Path, optional
        Path to the Stan model file. If None, uses the default BKT Stan model.
    compile_kwargs : dict, optional
        Additional keyword arguments for compiling the Stan model.

    Attributes
    ----------
    model_ : CmdStanModel
        The compiled Stan model.
    fit_ : CmdStanMCMC | CmdStanVB | CmdStanOptimize
        The fitted model after training.
    """
