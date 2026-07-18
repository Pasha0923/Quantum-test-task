import numpy as np
import evaluate
from transformers import (AutoModelForTokenClassification,DataCollatorForTokenClassification,Trainer,TrainingArguments,set_seed)
from preprocessing import (prepare_datasets,tokenizer)
from config import (MODEL_NAME,MODEL_OUTPUT_DIR,LABEL_LIST,LABEL2ID,ID2LABEL,BATCH_SIZE,LEARNING_RATE,NUM_EPOCHS,WEIGHT_DECAY,RANDOM_SEED)


metric = evaluate.load("seqeval")

def compute_metrics(eval_prediction):
    """
    Compute evaluation metrics for token classification.
    """

    predictions, labels = eval_prediction

    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):

        current_predictions = []
        current_labels = []

        for pred, lab in zip(prediction, label):

            if lab == -100:
                continue

            current_predictions.append(LABEL_LIST[pred])
            current_labels.append(LABEL_LIST[lab])

        true_predictions.append(current_predictions)
        true_labels.append(current_labels)

    results = metric.compute(
        predictions=true_predictions,
        references=true_labels,
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main():

    set_seed(RANDOM_SEED)

    MODEL_OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    train_dataset, validation_dataset = prepare_datasets()
    
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),

        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        save_total_limit=1,

        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)


if __name__ == "__main__":
    main()