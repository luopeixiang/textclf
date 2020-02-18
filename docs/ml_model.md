本文件根据[../textclf/config/ml_model.py](../textclf/config/ml_model.py)自动生成

### MLModelConfig

 无可设置的属性



### LogisticRegressionConfig

参考:https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

LogisticRegressionConfig继承MLModelConfig的所有属性，同时它还有以下属性：

 | Attribute name    | Type   | Default   | Description                                                                                                                                          |
|-------------------|--------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| penalty           | str    | "l2"      | penalty{‘l1’, ‘l2’, ‘elasticnet’, ‘none’} Used to specify the norm used in the penalization.                                                         |
| dual              | bool   | False     | Dual or primal formulation. Dual formulation is only implementedfor l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features. |
| tol               | float  | 1e-4      | Tolerance for stopping criteria.                                                                                                                     |
| C                 | float  | 1.0       | Inverse of regularization strengthmust be a positive float. Like in support vector machines, smaller values specify stronger regularization.         |
| fit_intercept     | bool   | True      | Specifies if a constant(a.k.a. bias or intercept) should be added to the decision function.                                                          |
| intercept_scaling | float  | 1         | Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True.                                                               |



### LinearSVMConfig

 无可设置的属性

