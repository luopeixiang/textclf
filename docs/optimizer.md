本文件根据[../textclf/config/optimizer.py](../textclf/config/optimizer.py)自动生成

### OptimizerConfig



OptimizerConfig有以下属性：

 | Attribute name   | Type   | Default   | Description                                                                                                |
|------------------|--------|-----------|------------------------------------------------------------------------------------------------------------|
| embedding_lr     | float  | 1e-3      | learning rate for embedding layer                                                                          |
| lr               | float  | 1e-3      | learning rate for other layer                                                                              |
| static_epoch     | int    | 0         | embedding层在前static_epoch将不会进行训练在static_epoch之后，embedding将会以embedding_lr作为学习率进行训练 |



### AdamConfig

Adam algorithm.
It has been proposed in Adam: A Method for Stochastic Optimization.

AdamConfig继承OptimizerConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type                | Default      | Description                                                                                              |
|------------------|---------------------|--------------|----------------------------------------------------------------------------------------------------------|
| betas            | Tuple[float, float] | (0.9, 0.999) | coefficients used for computing running averages of gradientand its square(default: (0.9, 0.999))        |
| eps              | float               | 1e-8         | term added to the denominator to improve numerical stability(default: 1e-8)                              |
| weight_decay     | float               | 0.           | weight decay(L2 penalty)(default: 0)                                                                     |
| amsgrad          | bool                | False        | whether to use the AMSGrad variant of this algorithm from the paperOn the Convergence of Adam and Beyond |



### AdadeltaConfig

Adadelta algorithm.
It has been proposed in ADADELTA: An Adaptive Learning Rate Method.

AdadeltaConfig继承OptimizerConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default   | Description                                                                         |
|------------------|--------|-----------|-------------------------------------------------------------------------------------|
| rho              | float  | 0.9       | coefficient used for computing a running average of squared gradients(default: 0.9) |
| eps              | float  | 1e-6      | term added to the denominator to improve numerical stability(default: 1e-6)         |
| weight_decay     | float  | 0.        | weight decay(L2 penalty)(default: 0)                                                |



### AdagradConfig

Adagrad algorithm.
It has been proposed in Adaptive Subgradient Methods for
Online Learning and Stochastic Optimization.

AdagradConfig继承OptimizerConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default   | Description                                                                   |
|------------------|--------|-----------|-------------------------------------------------------------------------------|
| lr_decay         | float  | 0.0       | learning rate decay (default: 0)                                              |
| weight_decay     | float  | 0.0       | weight decay (L2 penalty) (default: 0)                                        |
| eps              | float  | 1e-10     | term added to the denominator to improve numerical stability (default: 1e-10) |



### AdamWConfig

AdamW algorithm.
The original Adam algorithm was proposed in Adam:
A Method for Stochastic Optimization.
The AdamW variant was proposed in Decoupled Weight Decay Regularization.

AdamWConfig继承OptimizerConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type                | Default      | Description                                                                                                               |
|------------------|---------------------|--------------|---------------------------------------------------------------------------------------------------------------------------|
| betas            | Tuple[float, float] | (0.9, 0.999) | coefficients used for computing running averages of gradient and its square(default: (0.9, 0.999))                        |
| eps              | float               | 1e-8         | term added to the denominator to improve numerical stability(default: 1e-8)                                               |
| weight_decay     | float               | 1e-2         | weight decay coefficient(default: 1e-2)                                                                                   |
| amsgrad          | bool                | False        | whether to use the AMSGrad variant of this algorithm from the paperOn the  Convergence of Adam and Beyond(default: False) |



### AdamaxConfig

Adamax algorithm (a variant of Adam based on infinity norm).
It has been proposed in Adam: A Method for Stochastic Optimization.

AdamaxConfig继承OptimizerConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type                | Default      | Description                                                                                        |
|------------------|---------------------|--------------|----------------------------------------------------------------------------------------------------|
| betas            | Tuple[float, float] | (0.9, 0.999) | coefficients used for computing running averages of gradient and its square(default: (0.9, 0.999)) |
| eps              | float               | 1e-8         | term added to the denominator to improve numerical stability(default: 1e-8)                        |
| weight_decay     | float               | 1e-2         | weight decay coefficient(default: 1e-2)                                                            |



### ASGDConfig

Averaged Stochastic Gradient Descent.
It has been proposed in Acceleration of stochastic approximation by averaging.

ASGDConfig继承OptimizerConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default   | Description                                      |
|------------------|--------|-----------|--------------------------------------------------|
| lambd            | float  | 1e-4      | decay term (default: 1e-4)                       |
| alpha            | float  | 0.75      | power for eta update (default: 0.75)             |
| t0               | float  | 1e6       | point at which to start averaging (default: 1e6) |
| weight_decay     | float  | 0         | weight decay (L2 penalty) (default: 0)           |



### RMSpropConfig

RMSprop algorithm.
Proposed by G. Hinton in his course.

RMSpropConfig继承OptimizerConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default   | Description                                                                                        |
|------------------|--------|-----------|----------------------------------------------------------------------------------------------------|
| momentum         | float  | 0.0       | momentum factor(default: 0)                                                                        |
| alpha            | float  | 0.99      | smoothing constant(default: 0.99)                                                                  |
| eps              | float  | 1e-8      | term added to the denominator to improve numerical stability(default: 1e-8)                        |
| centered         | bool   | False     | if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance |
| weight_decay     | float  | 0.        | weight decay(L2 penalty)(default: 0)                                                               |



### RpropConfig

The resilient backpropagation algorithm.

RpropConfig继承OptimizerConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type                | Default    | Description                                                                                              |
|------------------|---------------------|------------|----------------------------------------------------------------------------------------------------------|
| etas             | Tuple[float, float] | (0.5, 1.2) | pair of (etaminus, etaplis), that are multiplicative increase and decrease factors (default: (0.5, 1.2)) |
| step_sizes       | Tuple[float, float] | (1e-6, 50) | a pair of minimal and maximal allowed step sizes (default: (1e-6, 50))                                   |



### SGDConfig

stochastic gradient descent (optionally with momentum).
Nesterov momentum is based on the formula from
On the importance of initialization and momentum in deep learning.

SGDConfig继承OptimizerConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default   | Description                                |
|------------------|--------|-----------|--------------------------------------------|
| momentum         | float  | 0.        | momentum factor (default: 0)               |
| weight_decay     | float  | 0.        | weight decay (L2 penalty) (default: 0)     |
| dampening        | float  | 0.        | dampening for momentum (default: 0)        |
| nesterov         | bool   | False     | enables Nesterov momentum (default: False) |

