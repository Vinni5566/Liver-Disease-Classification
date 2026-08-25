# Final model comparison and error analysis

## Scope

This package was created from the two saved evaluation metrics JSON artifacts and their
already-recorded confusion matrices. No dataset images, checkpoints, models, training,
or evaluation code were accessed or run.

## Results-section-ready summary

On the held-out evaluation, ResNet50 Stage 2 achieved higher accuracy (0.687 vs
0.667) and macro F1 (0.610 vs 0.577) than
EfficientNet-B0 under the final argmax decision rule. Both models had identical Malignant
recall (82.76%, 72/87). EfficientNet-B0 had slightly higher Malignant precision/F1
(0.791/0.809 vs
0.774/0.800) and
faster measured inference (30.7 vs 98.7
ms/image). The dominant error pattern for both models was Benign <-> Malignant confusion.

## Overall performance and selection rationale

ResNet50 has better accuracy (0.6871 vs 0.6667) and macro F1
(0.6098 vs 0.5767), making it the preferred option when balanced
performance under this fixed argmax operating rule is primary. This does not make
ResNet50 universally superior: EfficientNet-B0 has higher macro recall
(0.6259 vs 0.5925), higher Malignant precision/F1, and
substantially faster measured inference. Choice should remain dependent on intended
operating constraints and further validation.

## Per-class behaviour

- **Malignant:** Both models recalled 72 of 87 images (82.76%). EfficientNet-B0 had higher
  precision (0.7912 vs 0.7742)
  and F1 (0.8090 vs 0.8000).
- **Benign:** Benign classification was difficult, especially for EfficientNet-B0: recall
  0.25 vs 0.50 and F1
  0.317 vs 0.500.
- **Normal:** EfficientNet-B0 had much higher Normal recall (0.80 vs
  0.45) but lower precision (0.485 vs
  0.643).

## AUC, speed, and error patterns

EfficientNet-B0 has higher one-vs-rest AUC for Normal (0.939 vs
0.920), Benign (0.722 vs
0.719), and Malignant (0.866 vs
0.837). AUC measures ranking/discrimination across
thresholds, whereas accuracy and F1 reflect the selected argmax decision rule; higher AUC
alone therefore does not establish EfficientNet-B0 as superior for the reported rule.

The largest pairwise error was Benign <-> Malignant for both models (30 ResNet50; 28
EfficientNet-B0). Normal <-> Benign totals were 10 and 15; Normal <-> Malignant totals
were 6 for each. Malignant false negatives totalled 15/87 for both models. ResNet50 had
2 Malignant -> Normal and 13 Malignant -> Benign errors; EfficientNet-B0 had 6 and 9.
These clinically important missed-Malignant patterns should be examined explicitly.

Saved inference measurements were 98.7 ms/image for ResNet50
and 30.7 ms/image for EfficientNet-B0. Timing is
environment-dependent and should be re-measured for deployment.

## Limitations

This single held-out comparison has no confidence intervals, external cohort, calibration
assessment, reader study, or statistical significance testing. Neither model is clinically
validated, and these descriptive results do not support diagnostic or causal claims.
