from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer
from datasets import Dataset
import pandas as pd

from typing import Optional, Union

from .config import DEFAULT_CONFIGURATION


def multilabel_classification(example, tokenizer):
	'''
	'''
	return example

def entailment_based_classification(example, tokenizer):
	'''
	'''
	return example


class EmoBias:
	def __init__(
		self, 
		model_name: str, 
		problem_type: str = 'multi_label_classification',
		emotions: Optional[Union[str, list]] = '',
		source_language: Optional[Union[str, list]] = 'en',
		target_language_prompt: Optional[Union[bool, str]] = False,
		prompt_prefix: Optional[str] = None,
		output_dir: str = "outputs",
		learning_rate: float = 2e-5,
		per_device_train_batch_size: int = 3,
		per_device_eval_batch_size: int = 3,
		num_train_epochs: int = 2,
		weight_decay: float = 0.01,
		evaluation_strategy: str = "epoch",

	):
		params = {
			'id2label' : [],
			'label2id' : [],
			'num_labels' : 0,
			'problem_type' : ''
		}

		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **params)
		self.source_language = source_language
		self.target_language_prompt = target_language_prompt
		self.prompt_prefix = prompt_prefix
		self.output_dir = output_dir


	def train(self):
		training_args = TrainingArguments(
			output_dir=self.output_dir,
			learning_rate=2e-5,
			load_best_model_at_end=True
		)

		trainer = Trainer(
		   model=self.model,
		   args=training_args,
		   train_dataset=tokenized_dataset["train"],
		   eval_dataset=tokenized_dataset["test"],
		   tokenizer=tokenizer,	`
		   data_collator=data_collator,
		   compute_metrics=compute_metrics,
		)

		trainer.train()
