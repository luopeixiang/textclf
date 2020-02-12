import torch
from transformers import BertTokenizer

from .base_tester import Tester
from textclf.utils.raw_data import create_tokenizer
from textclf.utils.create import create_instance
from textclf.config import DLTesterConfig
from textclf.data.dictionary import Dictionary


class DLTester(Tester):
    """负责Deep Learning model的测试"""

    def __init__(self, config: DLTesterConfig):
        super().__init__(config)
        self.tokenizer = create_tokenizer(self.config.tokenizer)
        self.use_cuda = self.config.use_cuda and torch.cuda.is_available()

        print(f"Load checkpoint from {self.config.model_path}..")
        checkpoint = torch.load(self.config.model_path)
        self.model_conf, self.dictionary, self.label2id = \
            checkpoint["info_for_test"]
        self.model = create_instance(self.model_conf)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.classes = sorted(self.label2id, key=self.label2id.get)

    def _preprocess(self, text):
        text_tokenized = self.tokenizer(text)
        if isinstance(self.dictionary, Dictionary):
            text_processed = self.dictionary.tokens_to_tensor(
                text_tokenized, max_len=self.config.max_len
            )
            text_len = (text_processed != self.dictionary.pad()).sum()
        elif isinstance(self.dictionary, BertTokenizer):
            text_processed = torch.LongTensor(
                self.dictionary.encode(text_tokenized, add_special_tokens=True)[:-1])
            max_len = self.config.max_len
            pad_id = self.dictionary.pad_token_id
            if len(text_processed) >= max_len:
                text_processed = text_processed[:max_len]
            else:
                text_processed = torch.cat([
                    text_processed,
                    torch.ones(max_len-len(text_processed)).long()*pad_id
                ])
            text_len = (text_processed != pad_id).sum()

        if self.use_cuda:
            text_processed = text_processed.cuda()
            text_len = text_len.cuda()

        return text_processed.unsqueeze(0), text_len.unsqueeze(0)

    def predict_label(self, text):
        text_processed, text_len = self._preprocess(text)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(text_processed, text_len)
        label_id = torch.argmax(logits)
        return self.classes[label_id]

    def predict_prob(self, text):
        text_processed, text_len = self._preprocess(text)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(text_processed, text_len)[0]
        return torch.softmax(logits, dim=0).tolist()

    def get_all_labels(self):
        return self.classes
