"""Cztery warianty dostrajania HerBERTa do klasyfikacji binarnej."""

from peft import LoraConfig, PromptTuningConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForSequenceClassification

from config import CLASSIFIER_DROPOUT, DROPOUT, MODEL_NAME


METHODS = ["transfer", "full", "lora", "soft_prompt"]


def build_model(method="full", dropout=DROPOUT, classifier_dropout=CLASSIFIER_DROPOUT):
    """
    transfer    - zamrożona baza, trenowana tylko głowica klasyfikująca
    full        - pełny fine-tuning wszystkich ~124 mln wag
    lora        - adaptery LoRA na macierzach query/value (~0.24% wag)
    soft_prompt - prompt tuning, 20 wirtualnych tokenów (~0.014% wag)
    """
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        classifier_dropout=classifier_dropout,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=config, ignore_mismatched_sizes=True
    )

    if method == "transfer":
        for p in model.base_model.parameters():
            p.requires_grad = False
    elif method == "full":
        pass  # wszystkie wagi pozostają trenowalne
    elif method == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif method == "soft_prompt":
        peft_config = PromptTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=20)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        raise ValueError(f"Unknown method: {method}")

    return model
