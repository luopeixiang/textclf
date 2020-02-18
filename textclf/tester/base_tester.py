from abc import ABCMeta, abstractmethod

from textclf.utils.eval import evaluate


class Tester(metaclass=ABCMeta):
    """Trainer基类，规定MLTrainer、DLTrainer的接口"""

    def __init__(self, config):
        self.config = config

    def test(self):
        if self.config.interactive:
            self.interactive()
        else:
            self.test_file()

    def interactive(self):
        """交互式地对输入进行预测"""
        while True:
            text = input("Input(Enter q to quit) >")
            text = text.strip()
            if text == 'q':
                print("Exit!")
                break
            label = self.predict_label(text)
            print(f"Predicted Label: {label}")
            if self.config.predict_prob:
                prob = self.predict_prob(text)
                for label, p in zip(self.get_all_labels(), prob):
                    print(f"{label} : {p:.3f}")

    @abstractmethod
    def predict_label(self, text: str):
        """predict label for text"""
        pass

    @abstractmethod
    def predict_prob(self, text: str):
        """predict label prob for text"""
        pass

    @abstractmethod
    def get_all_labels(self):
        """按照ID返回label列表"""
        pass

    def test_file(self):
        """对给定的文件进行预测"""

        inp = open(self.config.input_file)
        out = open(self.config.out_file, 'w')

        has_target = self.config.has_target
        predict_prob = self.config.predict_prob
        write_badcase = (self.config.badcase_file is not None) and has_target
        if write_badcase:
            badcase = open(self.config.badcase_file, 'w')
        header = ["text", "predict"]
        if has_target:
            header.append("target")
        if predict_prob:
            header += self.get_all_labels()
        out.write('\t'.join(header)+'\n')
        if write_badcase:
            badcase.write('\t'.join(header)+'\n')

        predicts = []
        if has_target:
            targets = []

        for line in inp:
            if has_target:
                text, target = line.strip().split('\t')
                targets.append(target)
            else:
                text = line.strip()
            predict = self.predict_label(text)
            predicts.append(predict)

            out_fields = [text, predict]
            if has_target:
                out_fields.append(target)
            if predict_prob:
                out_fields += self.predict_prob(text)

            out_line = '\t'.join([str(x) for x in out_fields])+'\n'
            out.write(out_line)
            if write_badcase and target != predict:
                badcase.write(out_line)

        inp.close()
        out.close()
        print(f"Writing predicted labels to {self.config.out_file}")
        if write_badcase:
            print(f"Writing badcase to {self.config.badcase_file}")
            badcase.close()

        if self.config.has_target:
            # 输出混淆矩阵/ badcase等信息
            acc, report, confusion = evaluate(
                targets, predicts, self.get_all_labels())
            print(f"Acc in test file:{acc*100:.2f}%")
            print(f"Report:\n{report}")
            if self.config.print_confusion_mat:
                print(f"Confusion matrix:\n{confusion}")
