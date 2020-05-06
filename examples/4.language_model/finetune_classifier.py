from biome.text.api_new import Pipeline
from biome.text.api_new.configuration import TrainerConfiguration
from biome.text.api_new.helpers import yaml_to_dict
from biome.text.api_new.modules.heads import TextClassification

if __name__ == "__main__":
    # load an existing pre-trained model
    pipe = Pipeline.from_pretrained("configs/experiment_lm/model.tar.gz")

    labels = [
        "Allgemeinärzte",
        "Apotheken",
        "Architekturbüros",
        "Autowerkstätten",
        "Dienstleistungen",
        "Edv",
        "Elektriker",
        "Elektrotechnik",
        "Friseure",
        "Gebrauchtwagen",
        "Handelsvermittler Und -vertreter",
        "Hotels",
        "Maler",
        "Physiotherapie",
        "Restaurants",
        "Sanitärinstallationen",
        "Tiefbau",
        "Unternehmensberatungen",
        "Vereine",
        "Vermittlungen",
        "Versicherungsvermittler",
        "Werbeagenturen",
    ]

    pipe.set_head(TextClassification, pooler={"type": "boe"}, labels=labels)
    trainer = TrainerConfiguration(**yaml_to_dict("configs/trainer.yml"))
    trained_pl = pipe.train(
        output="text_classifier_fine_tuned",
        trainer=trainer,
        training="configs/train.data.yml",
        validation="configs/val.data.yml",
    )
