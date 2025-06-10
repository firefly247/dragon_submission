import re
from typing import List, Union

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "dragon_baseline" / "src"))

from dragon_baseline import DragonBaseline


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
        # for df in [self.df_train, self.df_val, self.df_test]:
        #     df[self.task.input_name] = df[self.task.input_name].map(self.custom_text_cleaning)


if __name__ == "__main__":
    DragonSubmission().process()
