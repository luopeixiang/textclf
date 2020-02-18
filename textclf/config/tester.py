
from .base import ConfigBase


class BaseTesterConfig(ConfigBase):
    # 分词的方法 应该和训练时的处理方法保持一致
    tokenizer: str = "char"

    # 是否通过交互的方式进行预测
    interactive: bool = False

    # 是否输出概率 是否输出每个标签的概率
    # 注意有些模型比如SVM是无法输出概率分布的
    predict_prob: bool = False

    # 输入文件，当interactive为True时，该字段无效
    input_file: str = "test.csv"
    # 输出文件：模型预测的结果将写入到out_file
    out_file: str = "predict.csv"

    # has_target参数指明inpiut_file中是否含有标签
    has_target: bool = True
    # 如果has_target为True, 可以选择是否将评估的结果保存到evaluate_file
    # 如果不为None，那么将badcase写入到badcase_file指定的文件
    badcase_file: str = None

    # 是否打印混淆矩阵，注意，当类别数比较多时，该矩阵可能无法正常在终端中显示
    print_confusion_mat: bool = False


class MLTesterConfig(BaseTesterConfig):
    model_path: str = "ckpts/model.joblib"
    vec_path: str = "ckpts/vectorizer.joblib"


class DLTesterConfig(BaseTesterConfig):
    use_cuda: bool = True
    model_path: str = "ckpts/best.pt"
    max_len: int = 10
