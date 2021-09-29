import logging
import os
from typing import List, Optional, Union
from transformers.data import DataProcessor, InputExample, InputFeatures, SingleSentenceClassificationProcessor
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    PreTrainedTokenizer,
    glue_compute_metrics,
    glue_output_modes,
    glue_processors
)
logger = logging.getLogger(__name__)

def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float]:
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length, padding=True, truncation=True, return_token_type_ids=True
    )

    features = []
    print(range(len(examples)))

    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:2]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features

class SentenceDataProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return NotImplementedError

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

# monkey-patch all glue classes to have test examples
def get_test_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

for task in glue_processors:

    processor = glue_processors[task]
    processor.get_test_examples = get_test_examples


class flProcessor(SentenceDataProcessor):
    def get_labels(self):
        labels = ['RS', 'S', 'Q', 'A', 'PF', 'ACK', 'RF', 'NF', 'C', 'H', 'DIR', 'RC', 'LF']
        # labels = ['A', 'ACK', 'C', 'DIR', 'H', 'LF', 'NF', 'PF', 'Q', 'RC', 'RF', 'S']

        return labels

class slProcessor(SentenceDataProcessor):
    def get_labels(self):
        labels = ['ACK', 'AEX', 'AR', 'AWH', 'AYN', 'C', 'DIR', 'E', 'FNU', 'FU',
                  'GRE', 'HI', 'I', 'LF', 'NF', 'O', 'OEX', 'PF', 'QD', 'QEX', 'QF',
                  'QI', 'QO', 'QP', 'QQ', 'QR', 'R', 'RC', 'RF', 'RFI']

        return labels

processors = glue_processors.copy()
processors.update(
    {"firstlevel":flProcessor, "secondlevel": slProcessor}
)
print(processors)
output_modes = glue_output_modes
output_modes.update(
    {"firstlevel":"classification", "secondlevel":"classification"}
)

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name in ["firstlevel", "secondlevel"]:
        return {"f1":f1_score(y_true=labels, y_pred=preds, average="weighted"), "acc": accuracy_score(y_true=labels, y_pred=preds)}
    elif task_name in glue_processors:
        return glue_compute_metrics(task_name, preds, labels)
    else:
        raise NotImplementedError

