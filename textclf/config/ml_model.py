"""各种模型的设置"""
from typing import Union, Optional, Dict

from .base import ConfigBase


class MLModelConfig(ConfigBase):
    pass


class LogisticRegressionConfig(MLModelConfig):
    """参考:https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"""
    # penalty{‘l1’, ‘l2’, ‘elasticnet’, ‘none’} Used to specify the norm used in the penalization.
    penalty: str = "l2"

    # Dual or primal formulation. Dual formulation is only implemented
    # for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.
    dual: bool = False

    # Tolerance for stopping criteria.
    tol: float = 1e-4

    # Inverse of regularization strength
    # must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
    C: float = 1.0

    # Specifies if a constant(a.k.a. bias or intercept) should be added to the decision function.
    fit_intercept: bool = True
    # Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True.
    intercept_scaling: float = 1

    # dict or ‘balanced’ or None, default="balanced"
    # Weights associated with classes in the form {class_label: weight}.
    # If not given, all classes are supposed to have weight one.
    # The “balanced” mode uses the values of y to automatically adjust weights
    # inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
    class_weight: Union[str, None, Dict[str, float]] = "balanced"

    # {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
    # Algorithm to use in the optimization problem.
    # For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
    solver: str = 'sag'

    # Maximum number of iterations taken for the solvers to converge.
    max_iter: int = 1000
    # multi_class{‘auto’, ‘ovr’, ‘multinomial’} =’auto’
    multi_class: str = 'ovr'
    # For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
    verbose: int = 0

    # The seed of the pseudo random number generator to use when shuffling the data.
    # If int, random_state is the seed used by the random number generator
    # If None, the random number generator is the RandomState instance used
    # by np.random. Used when solver == ‘sag’ or ‘liblinear’.
    random_state: int = None

    # Number of CPU cores used when parallelizing over classes if multi_class =’ovr’”. T
    n_jobs: int = None
    # The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1.
    l1_ratio: Optional[float] = None


class LinearSVMConfig(MLModelConfig):

    # dict or ‘balanced’ or None, default="balanced"
    # Weights associated with classes in the form {class_label: weight}.
    # If not given, all classes are supposed to have weight one.
    # The “balanced” mode uses the values of y to automatically adjust weights
    # inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
    class_weight: Union[str, None, Dict[str, float]] = "balanced"

    # The penalty (aka regularization term) to be used. Defaults to ‘l2’ which is the standard regularizer for linear
    # SVM models. ‘l1’ and ‘elasticnet’ might bring sparsity to the model (feature selection) not achievable with ‘l2’.
    penalty: str = 'l2'

    # Constant that multiplies the regularization term. Defaults to 0.0001.
    # Also used to compute learning_rate when set to ‘optimal’.
    alpha: float = 0.0001

    # The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio = 0 corresponds to
    # L2 penalty, l1_ratio = 1 to L1. Defaults to 0.15.
    l1_ratio: float = 0.15

    # Whether the intercept should be estimated or not. If False, the data is assumed to be already centered.
    fit_intercept: bool = True

    # The maximum number of passes over the training data(aka epochs).
    # It only impacts the behavior in the fit method, and not the partial_fit method.
    max_iter: int = 1000

    # The stopping criterion. If it is not None, the iterations will stop when(loss > best_loss - tol)
    # for n_iter_no_change consecutive epochs.
    tolfloat = 1e-3

    # Whether or not the training data should be shuffled after each epoch.
    shufflebool = True

    # The verbosity level.
    verboseint = 0

    # The number of CPUs to use to do the OVA(One Versus All, for multi-class problems) computation. None means 1
    # unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
    n_jobs: int = None

    # The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the
    # seed used by the random number generator
    # If RandomState instance, random_state is the random number generator
    # If None, the random number generator is the RandomState instance used by np.random.
    random_state: Optional[int] = None

    # Number of iterations with no improvement to wait before early stopping.
    n_iter_no_change: int = 5
