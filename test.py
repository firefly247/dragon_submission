from pathlib import Path

from dragon_eval import DragonEval

from process import DragonSubmission

if __name__ == "__main__":
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
        "Task109_Example_ml_ner-fold0",
    ]:
        DragonSubmission(
            input_path=Path(f"test-input/{job_name}"),
            output_path=Path(f"test-output/{job_name}"),
            workdir=Path(f"test-workdir/{job_name}"),
        ).process()

    # evaluate the predictions
    DragonEval(
        ground_truth_path=Path("test-ground-truth"),
        predictions_path=Path(f"test-output"),
        output_file=Path("test-output/metrics.json"),
        folds=[0]
    ).evaluate()

    print("Please check that all performances are above random guessing! For tasks 101-107, the performance should be above 0.7, for tasks 108-109 above 0.2.")
