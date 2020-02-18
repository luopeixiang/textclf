import os
import sys

import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer

from .base_trainer import Trainer
from textclf.config import DLTrainerConfig, BertEmbeddingLayerConfig, StaticEmbeddingLayerConfig
from textclf.utils.create import create_instance, create_optimizer, create_lr_scheduler
from textclf.utils.raw_data import load_raw_data
from textclf.data.loader import build_loader
from textclf.utils.training import cal_accuracy
from textclf.utils.config import get_instance_name


class DLTrainer(Trainer):
    """
    Base Trainer class that provide ways to
        1 Train deep model such as TextCNN, TextRCNN, TextRNN and so on,
            compute metrics against eval set and use the metrics for
            model selection.
        2 Test trained model, compute and publish metrics against a blind test set.
    """

    def __init__(self, config: DLTrainerConfig):
        super().__init__(config)
        self.use_cuda = self.config.use_cuda and torch.cuda.is_available()
        self.prepare()
        self.optimizer = create_optimizer(self.config.optimizer, self.model)
        self.lr_scheduler = create_lr_scheduler(self.config.scheduler, self.optimizer)


        self.config.criterion.use_cuda = self.use_cuda
        self.criterion = create_instance(self.config.criterion)

        self.best_acc = 0.0
        self.best_loss = np.inf
        self.static_epoch = self.config.optimizer.static_epoch
        self.log_interval = self.config.num_batch_to_print

        self.start_epoch = 1
        self.cur_epoch = 1
        if self.config.state_dict_file:
            self.load_state_dict(self.config.state_dict_file)
            print(f"Continue Training from epoch {self.start_epoch}!")

    def prepare(self):
        """prepare model and data for model input"""
        loader_config = self.config.data_loader
        raw_data = load_raw_data(loader_config.raw_data_path)
        self.config.model.classifier.output_size = len(raw_data.label2id)

        emb_conf = self.config.model.embedding_layer
        if isinstance(emb_conf, StaticEmbeddingLayerConfig):
            emb_conf.dictionary = raw_data.dictionary
            dictionary_or_tokenizer = raw_data.dictionary
        elif isinstance(emb_conf, BertEmbeddingLayerConfig):
            dictionary_or_tokenizer = BertTokenizer.from_pretrained(emb_conf.model_dir)

        print(f"Build model:\n{self.config.model}")
        self.model = create_instance(self.config.model)
        if self.use_cuda:
            self.model = self.model.cuda()
        print(f"Build data loader:\n{self.config.data_loader}")
        self.train_loader = build_loader(
            raw_data.train_pairs,
            dictionary_or_tokenizer,
            raw_data.label2id,
            loader_config
        )
        self.valid_loader = build_loader(
            raw_data.valid_pairs,
            dictionary_or_tokenizer,
            raw_data.label2id,
            loader_config
        )
        self.test_loader = build_loader(
            raw_data.test_pairs,
            dictionary_or_tokenizer,
            raw_data.label2id,
            loader_config
        )
        # config 中的embedding_layer 以及model用于在测试时初始化模型
        # dictionary 和label2id在测试的时候也需要用到
        self.info_for_test = (
            self.config.model,
            dictionary_or_tokenizer,
            raw_data.label2id
        )

    def train(self):
        for epoch in range(self.start_epoch, self.config.epochs+1):
            self.cur_epoch = epoch
            self.train_epoch()
            self.validate()

            if epoch > self.static_epoch:
                # print("Create new optimizer and scheduler!")
                self.optimizer = create_optimizer(self.config.optimizer, self.model, epoch)
                self.lr_scheduler = create_lr_scheduler(self.config.scheduler, self.optimizer)

            if self.config.save_ckpt_every_epoch:
                self.save_checkpoint(str(epoch))

        if self.config.do_eval:
            self.test()

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        for batch_id, batch in enumerate(self.train_loader, 1):
            loss, acc = self.run_step(batch, is_train=True)
            total_loss += loss
            total_acc += acc
            if batch_id % self.config.num_batch_to_print == 0:
                print(
                    f"Epoch {self.cur_epoch}/{self.config.epochs}\t"
                    f"Iter {batch_id}/{len(self.train_loader)}\t"
                    f"Loss {total_loss/self.log_interval:.4f}\t"
                    f"Accuracy {total_acc*100/self.log_interval:.2f}%\t"
                )
                total_loss = 0.0
                total_acc = 0.0

    def run_step(self, batch, is_train=True):
        text_tensor, text_lens, labels = batch
        if self.use_cuda:
            text_tensor = text_tensor.cuda()
            text_lens = text_lens.cuda()
            labels = labels.cuda()

        logits = self.model(text_tensor, text_lens)
        loss = self.criterion(logits, labels)
        acc = cal_accuracy(logits, labels)
        if is_train:
            max_norm = self.config.max_clip_norm
            if max_norm:
                clip_grad_norm_(self.model.parameters(), max_norm)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item(), acc

    def eval_dataloader(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batch = len(data_loader)
        with torch.no_grad():
            for batch in data_loader:
                loss, acc = self.run_step(batch, is_train=False)
                total_acc += acc
                total_loss += loss
        return total_loss/num_batch, total_acc/num_batch

    def test(self):
        if self.config.load_best_model_after_train:
            best_ckpt_path = os.path.join(self.config.ckpts_dir, "best.pt")
            self.load_state_dict(best_ckpt_path)
        test_loss, test_acc = self.eval_dataloader(self.test_loader)
        print(f"Test Loss: {test_loss:.4f}\tTest Acc: {test_acc*100:.2f}%")

    def validate(self):
        """eval model on validation dataset and save if find best model"""
        val_loss, val_acc = self.eval_dataloader(self.valid_loader)
        print(f"Validation Loss: {val_loss:.4f}\t"
              f"Validation Acc: {val_acc*100:.2f}%")

        find_best_model = False
        if val_loss < self.best_loss:
            print(f"Find best model with loss {val_loss:.4f}")
            self.best_loss = val_loss
            if self.config.score_method == "loss":
                find_best_model = True

        if val_acc > self.best_acc:
            print(f"Find best model with acc {val_acc:.4f}")
            self.best_acc = val_acc
            if self.config.score_method == "accuracy":
                find_best_model = True

        if find_best_model:
            self.save_checkpoint("best")
            self.best_epoch = self.cur_epoch
        else:
            self.check_early_stop()

        # learning rate schedule
        if self.lr_scheduler is not None:
            sname = get_instance_name(self.config.scheduler)
            if sname == "ReduceLROnPlateau":
                self.lr_scheduler.step(
                    val_loss if self.config.scheduler.mode == "min" else val_acc
                )
            else:
                self.lr_scheduler.step()

    def save_checkpoint(self, file_prefix):
        state_dict = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "best_acc": self.best_acc,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "info_for_test": self.info_for_test
        }
        os.makedirs(self.config.ckpts_dir, exist_ok=True)
        file_path = os.path.join(self.config.ckpts_dir, f"{file_prefix}.pt")
        print(f"Saving model to {file_path}...")
        torch.save(state_dict, file_path)

    def load_state_dict(self, path):
        """根据state文件初始化参数"""
        print(f"Loading checkpoint from {path}..")
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.best_loss = checkpoint["best_loss"]
        self.best_acc = checkpoint["best_acc"]
        sche_dict = checkpoint["lr_scheduler"]
        if sche_dict is not None:
            self.lr_scheduler.load_state_dict(sche_dict)

    def check_early_stop(self):
        # check early stop
        esa = self.config.early_stop_after
        if esa is not None and (self.cur_epoch - self.best_epoch) > esa:
            print("No optimization for a long time, auto stop training...")
            sys.exit()
