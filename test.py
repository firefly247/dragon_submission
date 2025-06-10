from pathlib import Path

from dragon_eval import DragonEval

from process import DragonSubmission

from dragon_baseline import ensemble_predictions

if __name__ == "__main__":

    train_and_eval = True

    if train_and_eval:
        # Note: to debug (outside of Docker), you can set the input and output paths.
        for job_name in [
            "Task101_Example_sl_bin_clf-fold0",
            "Task102_Example_sl_mc_clf-fold0",
            "Task103_Example_mednli-fold0",
            "Task104_Example_ml_bin_clf-fold0",
            "Task105_Example_ml_mc_clf-fold0",
            "Task106_Example_sl_reg-fold0",
            "Task107_Example_ml_reg-fold0",
            "Task108_Example_sl_ner-fold0",
            "Task109_Example_ml_ner-fold0"
        ]:
            input_path = Path(f"test-input/{job_name}")
            output_path_ensemble = Path(f"test-output/{job_name}_ensemble")
        
            # 모델별 workdir
            workdir1 = Path(f"test-workdir/{job_name}_model1")
            workdir2 = Path(f"test-workdir/{job_name}_model2")
            workdir3 = Path(f"test-workdir/{job_name}_model3")

            # 저장 경로 설정
            output_path_pred1 = Path(f"test-output/{job_name}_model1")
            output_path_pred2 = Path(f"test-output/{job_name}_model2")
            output_path_pred3 = Path(f"test-output/{job_name}_model3")

            # 모델 1
            model1 = DragonSubmission(input_path=input_path, output_path=output_path_pred1, workdir=workdir1, model_name="joeranbosma/dragon-roberta-base-mixed-domain")
            model1.load()
            model1.validate()
            model1.analyze()
            model1.preprocess()
            model1.train()
            pred1 = model1.predict(df=model1.df_test)
            model1.save(pred1)

            # 모델 2
            model2 = DragonSubmission(input_path=input_path, output_path=output_path_pred2, workdir=workdir2, model_name="joeranbosma/dragon-bert-base-mixed-domain")
            model2.load()
            model2.validate()
            model2.analyze()
            model2.preprocess()
            model2.train()
            pred2 = model2.predict(df=model2.df_test)
            model2.save(pred2)

            # 모델 3
            model3 = DragonSubmission(input_path=input_path, output_path=output_path_pred3, workdir=workdir3, model_name="joeranbosma/dragon-roberta-base-domain-specific")
            model3.load()
            model3.validate()
            model3.analyze()
            model3.preprocess()
            model3.train()
            pred3 = model3.predict(df=model3.df_test)
            model3.save(pred3)

            # 앙상블 예측
            ensemble_pred = ensemble_predictions([pred1, pred2, pred3], model1.task.target.problem_type)
            model_ensemble = DragonSubmission(input_path=input_path, output_path = output_path_ensemble)
            test_predictions_ensemble_path = output_path_ensemble / "nlp-predictions-dataset.json"
            test_predictions_ensemble_path.parent.mkdir(parents=True, exist_ok=True)
            ensemble_pred.to_json(test_predictions_ensemble_path, orient="records")   

            # DragonSubmission(
            #     input_path=Path(f"test-input/{job_name}"),
            #     output_path=Path(f"test-output/{job_name}"),
            #     workdir=Path(f"test-workdir/{job_name}"),
            # ).process()

    # DragonEval(
    #     ground_truth_path=Path("test-ground-truth"),
    #     predictions_path=Path(f"test-output"),
    #     output_file=Path("test-output/metrics.json"),
    #     folds=[0]
    # ).evaluate()

    DragonEval(
        ground_truth_path=Path("test-ground-truth"),
        predictions_path=Path(f"test-output"),
        output_file=Path("test-output/metrics_ensemble.json"),
        folds=[0]
    ).evaluate()

    print("Please check that all performances are above random guessing! For tasks 101-107, the performance should be above 0.7, for tasks 108-109 above 0.2.")
