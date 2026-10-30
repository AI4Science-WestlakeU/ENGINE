# CoLA metrics in the current code

ENGINE's trainer (`trainer/train_loop_evvms.py`) selects Matthews correlation coefficient automatically when the task is `cola`. It collects predictions and labels across the validation batches and calls `sklearn.metrics.matthews_corrcoef`.

- `val_mcc` stores the raw MCC in [-1, 1]. Multiply by 100 for the scale used in the paper.
- Other tasks log `val_acc` as accuracy in percent.
- `model/glue_transformer.py` supports `return_predictions=True` for evaluation.
- The current command-line interface has no `--eval_metric` option.
- The separate baseline trainer (`trainer/train_loop.py`) still evaluates accuracy.

## Validation data

`--samplesize_validation` controls the validation subset. The supplied CoLA scripts currently request 256 examples. The GLUE validation loader uses `drop_last=True`, so an incomplete final batch is omitted. Increasing the subset to 1043 alone therefore does not evaluate all 1043 examples when the batch size is 64. For an evaluation over every example, the validation loader must retain the final batch.

The five-seed MCC means in the paper are aggregate reported results. This repository does not include per-step MCC exports or per-seed prediction files. Existing accuracy curves cannot be converted into MCC curves from those aggregate means.
