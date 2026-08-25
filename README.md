# AI-Assisted Liver Disease Classification from Ultrasound Images

An academic research prototype for three-class classification of liver ultrasound images as **Normal**, **Benign**, or **Malignant**. The project compares ImageNet-pretrained ResNet50 and EfficientNet-B0 models, uses Grad-CAM for model-attribution visualization, and provides a Streamlit demonstration interface. It is not a clinical diagnostic device.

## Dataset and fixed split

The repository uses the included AUL ultrasound liver dataset (735 images):

| Class | Label | Images |
|---|---:|---:|
| Normal | 0 | 100 |
| Benign | 1 | 200 |
| Malignant | 2 | 435 |

The fixed, stratified split is recorded in `dataset/development_split.json`. Train and validation selections are drawn only from the 588-image development split; the 147-image test set is held out for final evaluation.

| Split | Normal | Benign | Malignant | Total |
|---|---:|---:|---:|---:|
| Train | 64 | 128 | 278 | 470 |
| Validation | 16 | 32 | 70 | 118 |
| Held-out test | 20 | 40 | 87 | 147 |

The class order is always `Normal = 0`, `Benign = 1`, and `Malignant = 2`.

## Methodology

Both completed full-frame models use ImageNet-pretrained backbones: ResNet50 and EfficientNet-B0. Inputs are resized to 224 x 224 and normalized with ImageNet statistics. The repository's training augmentation is applied only to training images. Class imbalance is handled with `WeightedRandomSampler`; optimization uses Adam and `ReduceLROnPlateau`. BatchNorm running statistics are frozen, checkpoint selection is based on validation loss, and early stopping uses patience 10.

Training and model selection use only the fixed 470-image training subset and 118-image validation subset. The held-out test set is not used during training or selection.

## Staged transfer learning

Stage 1 trains the classification head with the backbone and BatchNorm frozen, learning rate `1e-3`, 15 epochs, batch size 32, and `WeightedRandomSampler`.

Stage 2 initializes from the corresponding Stage 1 best checkpoint, unfreezes the backbone while retaining frozen BatchNorm, and uses learning rate `1e-6`, a maximum of 30 epochs, batch size 32, `WeightedRandomSampler`, and validation-loss early stopping.

### ResNet50

```powershell
python train.py --model resnet50 --experiment-name staged_transfer/stage1_head_only --freeze-backbone --freeze-batchnorm --learning-rate 1e-3 --epochs 15 --batch-size 32 --imbalance-strategy sampler
python train.py --model resnet50 --experiment-name staged_transfer/stage2_finetune --resume-checkpoint models/resnet50/staged_transfer/stage1_head_only/best_model.pt --freeze-batchnorm --learning-rate 1e-6 --epochs 30 --batch-size 32 --imbalance-strategy sampler
```

### EfficientNet-B0

```powershell
python train.py --model efficientnet_b0 --experiment-name staged_transfer/stage1_head_only --freeze-backbone --freeze-batchnorm --learning-rate 1e-3 --epochs 15 --batch-size 32 --imbalance-strategy sampler
python train.py --model efficientnet_b0 --experiment-name staged_transfer/stage2_finetune --resume-checkpoint models/efficientnet_b0/staged_transfer/stage1_head_only/best_model.pt --freeze-batchnorm --learning-rate 1e-6 --epochs 30 --batch-size 32 --imbalance-strategy sampler
```

Final full-frame checkpoints:

- ResNet50: `models/resnet50/staged_transfer/stage2_finetune/best_model.pt`
- EfficientNet-B0: `models/efficientnet_b0/staged_transfer/stage2_finetune/best_model.pt`

## Held-out evaluation

Use the final Stage-2 checkpoints explicitly. Do not use the default evaluation paths for these completed staged experiments.

```powershell
python evaluate.py --model resnet50 --checkpoint models/resnet50/staged_transfer/stage2_finetune/best_model.pt
python evaluate.py --model efficientnet_b0 --checkpoint models/efficientnet_b0/staged_transfer/stage2_finetune/best_model.pt
```

Evaluation reports accuracy, macro and per-class precision/recall/F1, one-vs-rest ROC/AUC, a 3 x 3 confusion matrix, and inference time.

## Final full-frame results

The following results are from the single held-out 147-image test set.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Inference time |
|---|---:|---:|---:|---:|---:|
| ResNet50 | 0.6871 | 0.6390 | 0.5925 | 0.6098 | 98.7 ms/image |
| EfficientNet-B0 | 0.6667 | 0.5703 | 0.6259 | 0.5767 | 30.7 ms/image |

| Model | Malignant precision | Malignant recall | Malignant F1 |
|---|---:|---:|---:|
| ResNet50 | 0.7742 | 0.8276 | 0.8000 |
| EfficientNet-B0 | 0.7912 | 0.8276 | 0.8090 |

ResNet50 is the selected final model because it achieved stronger held-out accuracy and macro F1 under the final argmax decision rule. Both models achieved the same Malignant recall (0.8276). EfficientNet-B0 had faster measured CPU inference and slightly higher Malignant precision/F1. The dominant error pattern for both models was Benign <-> Malignant confusion.

### Confusion matrices

Rows are true classes and columns are predicted classes in the order Normal, Benign, Malignant.

```text
ResNet50
[[ 9,  7,  4],
 [ 3, 20, 17],
 [ 2, 13, 72]]

EfficientNet-B0
[[16,  4,  0],
 [11, 10, 19],
 [ 6,  9, 72]]
```

EfficientNet-B0 had higher one-vs-rest AUCs in the saved evaluation artifacts, but ROC/AUC describes ranking across thresholds. It does not by itself establish superior final argmax classification performance, which is reflected by the reported accuracy and F1 values.

## Grad-CAM

Grad-CAM is implemented manually with PyTorch forward and backward hooks in `utils/gradcam.py` and is architecture-agnostic. The resolved target layers are:

- ResNet50: `layer4.2.conv3`
- EfficientNet-B0: `features.8.0`

Examples were generated from the fixed validation data only. Grad-CAM highlights image regions contributing to a model prediction; it is a model-attribution visualization and does not establish medical causality.

## Streamlit application

Run the application with:

```powershell
streamlit run app/app.py
```

The application loads only the final full-frame Stage-2 checkpoints listed above. It supports ResNet50 and EfficientNet-B0, image upload, predicted class, confidence, class probabilities, a configurable manual-review threshold (70% by default), and a Grad-CAM overlay. The interface states: **Academic research prototype — not a clinical diagnostic device.**

## Limitations

This is an experimental research prototype using a relatively small, class-imbalanced dataset. The held-out test set comes from the same dataset rather than an external cohort. External validation, clinical validation, calibration analysis, reader studies, and statistical comparison were not performed. The results do not support clinical diagnostic claims, and Grad-CAM should not be interpreted as a medical causal explanation.
