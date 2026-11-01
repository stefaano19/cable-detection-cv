# Cable Detection with Mask R-CNN

An instance-segmentation project for detecting cable-shaped objects and estimating a dominant line from each predicted mask. The implementation is currently delivered as a Kaggle notebook rather than a packaged training or inference application.

The notebook builds on Torchvision Mask R-CNN with a ResNet-50/FPN backbone, fine-tunes the box and mask heads for a cable class, and post-processes masks into line parameters (`rho`, `theta`) using morphological thinning and the Hough transform.

> **Reproducibility status:** the notebook records no verified model metrics. Its saved execution output includes a failed Torch–NumPy interoperability check, and the current data augmentation and category-ID handling need correction before reported scores should be treated as reliable. Details are in [Limitations](#limitations-and-open-work).

## What is in this repository

- [`notebookb85dae4a5c.ipynb`](notebookb85dae4a5c.ipynb): data loading, training, validation, line extraction, and test-set prediction.
- [`Presentazione.pptx`](Presentazione.pptx): project presentation.
- `progetto_263703`: an additional extensionless asset; its format and role are not documented.

## Model and pipeline

1. **Input:** COCO-style images and annotations. The notebook reads training images and `train.json` from `/kaggle/input/progetto/progetto_cv_2025_2026/train`. It expects a `segmentation` field and uses `polar_coordinates` (`rho`, `theta`) for line evaluation.
2. **Dataset preparation:** invalid or very small boxes are filtered. Remaining annotations are converted to binary masks and the cable class is encoded as label `1`.
3. **Model:** `torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")`, with the box and mask predictors replaced for background plus cable. The notebook chooses CUDA when available.
4. **Training:** AdamW at `1e-4`; up to 80 epochs; validation every epoch; `ReduceLROnPlateau` in max mode (factor `0.5`, patience `3`, minimum learning rate `1e-6`); early stopping after 10 epochs without improvement. Augmentation changes from horizontal/vertical flips, ±5° rotation and color jitter, to horizontal flips and ±2° rotation, then no augmentation at epochs 30 and 55.
5. **Line extraction:** prediction masks are thresholded, filtered, thinned, and passed to `cv2.HoughLines`. Test inference filters candidates by score, area, and elongation, clusters by angle, and keeps one highest-scoring candidate from the largest angle cluster per image.

These settings describe the notebook as checked in; they are not a claim that the full run completed successfully.

## Data and split

The dataset is not included in this repository. Notebook metadata references Kaggle dataset inputs, and the code expects a Kaggle-mounted directory tree. Use only dataset versions you are authorized to access and check their terms before training or redistribution.

The notebook creates a single train/validation split from the training dataset: 15% validation, selected with a Torch generator seeded with `42` (the saved output reports 716 training and 126 validation samples). It then builds validation COCO annotations from those training annotations. This is a reproducible split for the same dataset ordering, but it is not cross-validation and does not provide an independent test metric. The notebook also has a test-inference path; its test annotation path uses a different Kaggle input root (`/kaggle/input/progetto-cv-25-26/...`) from the training root, so paths must be reconciled in the Kaggle environment.

## Evaluation

Validation uses COCO segmentation evaluation and reports AP at IoU 0.50 and AR at IoU 0.50. The notebook’s custom score is:

`LDS = AP50 + AR50 + 2 × mean(exp(-0.12 × angular_distance))`

The angle term is computed from greedily matched masks; normalized `rho` differences are also calculated but are not included in the LDS formula.

| Result | Reported value |
| --- | --- |
| Segmentation AP50 | Not reported / not verified |
| Segmentation AR50 | Not reported / not verified |
| Normalized `rho` difference | Not reported / not verified |
| Angular consistency term | Not reported / not verified |
| LDS | Not reported / not verified |
| Test-set metric | Not reported / not verified |

No AP50, latency, or other benchmark number is claimed here. The notebook’s saved outputs show setup and data-loading activity, but do not establish a successful evaluated model run.

## Reproducing the notebook

1. Open the [Kaggle notebook](https://www.kaggle.com/code/stefanobrusco/notebookb85dae4a5c?scriptVersionId=286285702) in an account and runtime with authorized access to its referenced dataset inputs.
2. Use a Python 3.11 GPU runtime if available. The notebook installs Torch/Torchvision/Torchaudio `2.1.2/0.16.2/2.1.2`, NumPy `1.26.4`, `pycocotools`, `opencv-python-headless`, and `shapely`.
3. Before training, reconcile the hard-coded training and test dataset paths with the inputs mounted in your runtime. Do not copy private data into this repository.
4. Resolve the dependency conflict before running the notebook end-to-end: after installing OpenCV, the saved run has NumPy `2.2.6` alongside Torch `2.1.2`; the notebook’s `torch.from_numpy` check fails with “Numpy is not available.” Pin a mutually compatible NumPy/OpenCV/Torch stack, restart the runtime, and rerun the import/interoperability check before proceeding.
5. Correct the augmentation and evaluation issues listed below, then execute cells in order and record the dataset version, split, package versions, checkpoint selection, and metrics for any new result.

There is no standalone CLI, `requirements.txt`, or local dataset configuration in this repository, so cloning the repository alone is not sufficient to reproduce training.

## Artifacts and weights

The notebook writes artifacts to Kaggle’s writable `/kaggle/working` directory:

- `maskrcnn_cable_ckpt.pth`: resumable checkpoint containing model and optimizer state, epoch, and best LDS.
- `maskrcnn_cable_best.pth`: best model state dictionary according to the notebook’s LDS.
- `maskrcnn_cable_final.pth`: final exported state dictionary after loading the best checkpoint.
- `gt_val_patched.json`, `val_preds_epoch_<epoch>.json`, and `predictions.json`: patched validation annotations, per-epoch validation predictions, and test predictions.

These generated weights and prediction files are not included here. If sharing a checkpoint, publish it through an approved artifact location with its source run, dataset/version provenance, package versions, and applicable usage terms; do not commit private data, credentials, or weights without authorization.

## Limitations and open work

- **Augmentation does not transform targets.** The notebook applies flips and rotations to the image with `torchvision.transforms`, while leaving the corresponding boxes and masks unchanged. This can misalign training labels. Use transforms that jointly update image, boxes, and masks, then rerun training before relying on model quality.
- **COCO category IDs are inconsistent.** The patched validation annotations use category ID `1`, while validation and test prediction records use category ID `0`. The evaluation helper also leaves its class filter disabled by default. Align prediction and ground-truth IDs and verify COCO evaluation on a small known example before interpreting AP/AR or LDS.
- **Line matching is permissive.** The custom analysis contains an IoU-threshold check commented out and accepts any positive-overlap greedy match. It also computes `rho` error without including it in LDS. Treat LDS as an experimental notebook metric, not a standardized benchmark.
- **Environment setup is not currently clean.** The saved notebook output documents a NumPy/Torch incompatibility after dependency installation; the recorded Torch–NumPy check fails. No clean end-to-end execution is established by the checked-in outputs.
- **Data and test setup are external.** Dataset access and terms are not verified here, and the training and test paths differ. The split uses one seed and one partition; no independent benchmark or latency measurement is reported.
- **Single-class scope.** The model and post-processing are configured for cable detection and a dominant line estimate; broader object classes and robustness across environments are not demonstrated.

## Contributions

Contributions that improve target-aware augmentation, dataset path configuration, category-consistent evaluation, dependency pinning, and documented result reporting are especially useful. For any proposed performance claim, include the exact dataset version/split, evaluation procedure, and measured output; do not infer missing numbers from the notebook configuration.
