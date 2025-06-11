import re
from typing import List, Union
import sys
import pandas as pd
import numpy as np
# from pathlib import Path
# sys.path.insert(str(Path(__file__).parent / "dragon_baseline" / "src"))

from dragon_baseline import DragonBaseline
from dragon_baseline.nlp_algorithm import ProblemType
from collections import Counter

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


def ensemble_predictions(
        pred_list: list[pd.DataFrame],
        problem_type: ProblemType
    ) -> pd.DataFrame:
    
    df_base = pred_list[0].copy()
    key = df_base.columns[-1]  # prediction column name

    if problem_type == ProblemType.SINGLE_LABEL_BINARY_CLASSIFICATION:
        # 평균 확률 → threshold 적용
        df_base[key] = sum(df[key] for df in pred_list) / len(pred_list)

    elif problem_type == ProblemType.MULTI_LABEL_BINARY_CLASSIFICATION:
        # sigmoid 결과 평균만 적용 (threshold는 나중에 적용)
        df_base[key] = [
            np.mean([df[key].iloc[i] for df in pred_list], axis=0).tolist()
            for i in range(len(df_base))
        ]

    elif problem_type == ProblemType.SINGLE_LABEL_MULTI_CLASS_CLASSIFICATION:
        # Hard voting (최빈값)
        df_base[key] = [
            pd.Series([df[key].iloc[i] for df in pred_list]).mode().iloc[0] # 최빈값 추출
            for i in range(len(df_base))
        ]

    elif problem_type == ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION:
        
        def majority_vote_labels(label_lists):
            top_k = len(label_lists[0])  # 첫 번째 모델의 예측 라벨 수
            # 모든 라벨 합쳐서 가장 많이 등장한 n개 선택
            flat_labels = [label for labels in label_lists for label in labels]
            counter = Counter(flat_labels)
            # 등장 횟수 순으로 정렬 후 top-k 반환 (예: 상위 3개)
            most_common = [label for label, _ in counter.most_common(top_k)]

            # 만약 top_k보다 적게 등장한 경우, 가장 마지막 라벨을 채워 넣음
            while len(most_common) < top_k:
                most_common.append(most_common[-1])

            return most_common

        df_base[key] = [
            majority_vote_labels([df[key].iloc[i] for df in pred_list])
            for i in range(len(df_base))
        ]

    elif problem_type in [
        ProblemType.SINGLE_LABEL_REGRESSION,
        ProblemType.MULTI_LABEL_REGRESSION
    ]:
        # 값 평균
        df_base[key] = sum(df[key] for df in pred_list) / len(pred_list)

    elif problem_type in [
        ProblemType.SINGLE_LABEL_NER,
        ProblemType.MULTI_LABEL_NER
    ]:
        # Token 단위 Hard Voting
        def majority_vote_tokenwise(token_lists):
            result = []
            for tokens in zip(*token_lists):  # i번째 토큰들을 모아서
                result.append(pd.Series(tokens).mode().iloc[0])
            return result

        df_base[key] = [
            majority_vote_tokenwise([df[key].iloc[i] for df in pred_list])
            for i in range(len(df_base))
        ]
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    return df_base

if __name__ == "__main__":
    # DragonSubmission().process()

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
    model2 = DragonSubmission(model_name="joeranbosma/dragon-bert-base-mixed-domain")
    model2.load()
    model2.validate()
    model2.analyze()
    model2.preprocess()
    model2.train()
    pred2 = model2.predict(df=model2.df_test)
    # model2.save(pred2)

    # 모델 3
    model3 = DragonSubmission(model_name="joeranbosma/dragon-roberta-base-mixed-domain")
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
    model_ensemble.save(ensemble_pred)
    model_ensemble.verify_predictions()