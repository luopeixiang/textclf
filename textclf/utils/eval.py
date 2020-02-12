from sklearn import metrics
from tabulate import tabulate

from typing import List


def evaluate(targets: List, predicts: List, class_list: List, digits=4):
    acc = metrics.accuracy_score(targets, predicts)
    report = metrics.classification_report(
        targets, predicts, target_names=class_list, digits=digits)

    confusion = metrics.confusion_matrix(targets, predicts)
    headers = [""] + class_list
    table = [[class_]+list(row) for class_, row in zip(class_list, confusion)]
    confusion_table = tabulate(table, headers, tablefmt="grid")
    return acc, report, confusion_table
