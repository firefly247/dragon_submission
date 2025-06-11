import re
from typing import List, Union
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "dragon_baseline" / "src"))

from dragon_baseline import DragonBaseline, ensemble_predictions

class DragonSubmission(DragonBaseline):
    def __init__(
        self,
        model_name: str = "joeranbosma/dragon-bert-base-mixed-domain",
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 2,
        gradient_checkpointing: bool = False,
        max_seq_length: int = 512,
        learning_rate: float = 1e-5,
        num_train_epochs: int = 5,
        **kwargs
    ):
        # Example of how to adapt the DRAGON baseline to use a different model
        """
        Adapt the DRAGON baseline to use the joeranbosma/dragon-roberta-base-mixed-domain model.
        Note: when changing the model, update the Dockerfile to pre-download that model.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs

    def custom_text_cleaning(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Perform custom text cleaning on the input text.

        Args:
            text (Union[str, List[str]]): The input text to be cleaned. It can be a string or a list of strings.

        Returns:
            Union[str, List[str]]: The cleaned text. If the input is a string, the cleaned string is returned.
            If the input is a list of strings, a list of cleaned strings is returned.

        """
        if isinstance(text, str):
            # Remove HTML tags and URLs:
            text = re.sub(r"<.*?>", "", text)
            text = re.sub(r"http\S+", "", text)

            return text
        else:
            # If text is a list, apply the function to each element
            return [self.custom_text_cleaning(t) for t in text]

    def preprocess(self):
        # Example of how to adapt the DRAGON baseline to use a different preprocessing function
        super().preprocess()

        # Uncomment the following lines to use the custom_text_cleaning function
        for df in [self.df_train, self.df_val, self.df_test]:
            df[self.task.input_name] = df[self.task.input_name].map(self.custom_text_cleaning)

if __name__ == "__main__":
    # 모델 1
    model1 = DragonSubmission(model_name="joeranbosma/dragon-roberta-large-domain-specific")
    model1.load()
    model1.validate()
    model1.analyze()
    model1.preprocess()
    model1.train()
    pred1 = model1.predict(df=model1.df_test)
    # model1.save(pred1)

    # 모델 2
    model2 = DragonSubmission(model_name="joeranbosma/dragon-roberta-large-mixed-domain")
    model2.load()
    model2.validate()
    model2.analyze()
    model2.preprocess()
    model2.train()
    pred2 = model2.predict(df=model2.df_test)
    # model2.save(pred2)

    # 모델 3
    model3 = DragonSubmission(model_name="joeranbosma/dragon-longformer-large-domain-specific")
    model3.load()
    model3.validate()
    model3.analyze()
    model3.preprocess()
    model3.train()
    pred3 = model3.predict(df=model3.df_test)
    # model3.save(pred3)

    # 앙상블
    ensemble_pred = ensemble_predictions([pred1, pred2, pred3], model1.task.target.problem_type)
    model_ensemble = DragonSubmission()
    model_ensemble.task = model1.task
    model_ensemble.save(ensemble_pred)
    model_ensemble.verify_predictions()