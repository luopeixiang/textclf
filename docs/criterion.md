本文件根据[../textclf/config/criterion.py](../textclf/config/criterion.py)自动生成

### CriterionConfig



CriterionConfig有以下属性：

 | Attribute name   | Type           | Default   | Description             |
|------------------|----------------|-----------|-------------------------|
| use_cuda         | Optional[bool] | None      | 是否使用GPU，运行时确定 |



### CrossEntropyLossConfig



CrossEntropyLossConfig继承CriterionConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type                  | Default   | Description                                                                                                                                                                                                                                         |
|------------------|-----------------------|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| weight           | Optional[List[float]] | None      | a manual rescaling weight given to each class. If given, has to be a Tensor of size C                                                                                                                                                               |
| reduction        | str                   | 'mean'    | Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.'none': no reduction will be applied,'mean': the sum of the output will be divided by the number of elements in the output,'sum': the output will be summed.Default: 'mean' |
| label_smooth_eps | Optional[float]       | None      | 是否应用label smooth                                                                                                                                                                                                                                |



### FocalLossConfig

Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002

FocalLossConfig继承CriterionConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default   | Description   |
|------------------|--------|-----------|---------------|
| reduction        | str    | "mean"    |               |

