from biome.text.data import DataSource
import pandas as pd
from biome.text import Pipeline
from biome.text.configuration import TrainerConfiguration

if __name__ == "__main__":
    train_ds = DataSource(source="/home/david/recognai/projects/CANTEMIST/cantemist/data/NER_david/train_wo_align_test.json", lines=True, orient="records")
    valid_ds = DataSource(source="/home/david/recognai/projects/CANTEMIST/cantemist/data/NER_david/dev1_wo_align.json", lines=True, orient="records")
    # train_ds.mapping = {"text": "text_org", "label": "file"}

    df = valid_ds.to_dataframe().compute()
    labels_total = df.labels.sum()
    pd.Series(labels_total).value_counts()

    pipeline_dict = {
        "name": "candemist-ner-distilmbert",
        "features": {
            "transformers": {
                # "model_name": 'dccuchile/bert-base-spanish-wwm-cased',  # does only work with 'trainable': False
                "model_name": "distilbert-base-multilingual-cased",  # does only work with 'trainable': False
                # "model_name": "distilroberta-base",  # DOES work with 'trainable': False and True
                "trainable": True,
                # "trainable": False,
            }
        },
        "head": {
            "type": "TokenClassification",
            # "type": "TextClassification",
            "labels": list(set(labels_total)),
            # "labels": ["cc_onco1.txt"],
        },
    }

    pl = Pipeline.from_config(pipeline_dict)

    trainer_config = TrainerConfiguration(
        optimizer={
            "type": "adam",
            "lr": 0.0000001,
        },
        batch_size=1,
        num_epochs=20,
        cuda_device=-1,
    )

    pl.train(
        output="test_output",
        training=train_ds,
        trainer=trainer_config
    )
