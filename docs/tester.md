本文件根据[../textclf/config/tester.py](../textclf/config/tester.py)自动生成

### BaseTesterConfig



BaseTesterConfig有以下属性：

 | Attribute name      | Type   | Default       | Description                                                                                                              |
|---------------------|--------|---------------|--------------------------------------------------------------------------------------------------------------------------|
| tokenizer           | str    | "char"        | 分词的方法 应该和训练时的处理方法保持一致                                                                                |
| interactive         | bool   | False         | 是否通过交互的方式进行预测                                                                                               |
| predict_prob        | bool   | False         | 是否输出概率 是否输出每个标签的概率注意有些模型比如SVM是无法输出概率分布的                                               |
| input_file          | str    | "test.csv"    | 输入文件，当interactive为True时，该字段无效                                                                              |
| out_file            | str    | "predict.csv" | 输出文件：模型预测的结果将写入到out_file                                                                                 |
| has_target          | bool   | True          | has_target参数指明inpiut_file中是否含有标签                                                                              |
| badcase_file        | str    | None          | 如果has_target为True, 可以选择是否将评估的结果保存到evaluate_file如果不为None，那么将badcase写入到badcase_file指定的文件 |
| print_confusion_mat | bool   | False         | 是否打印混淆矩阵，注意，当类别数比较多时，该矩阵可能无法正常在终端中显示                                                 |



### MLTesterConfig



MLTesterConfig继承BaseTesterConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default                   | Description   |
|------------------|--------|---------------------------|---------------|
| model_path       | str    | "ckpts/model.joblib"      |               |
| vec_path         | str    | "ckpts/vectorizer.joblib" |               |



### DLTesterConfig



DLTesterConfig继承BaseTesterConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default         | Description   |
|------------------|--------|-----------------|---------------|
| use_cuda         | bool   | True            |               |
| model_path       | str    | "ckpts/best.pt" |               |
| max_len          | int    | 10              |               |

