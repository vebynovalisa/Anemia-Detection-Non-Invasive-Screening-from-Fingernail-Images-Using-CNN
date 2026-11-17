# Anemia Detection from Fingernail RGB Images Using CNN

A research project on non-invasive anemia screening from fingernail RGB images. Three transfer-learning CNN architectures (DenseNet169, InceptionV3, Xception) are trained and compared on a combined public dataset of 4,466 images.

## Team & Contributions
A team project (Group 3) for the Final Project / Assurance of Learning (AOL) assignment of the Research Methodology in Computer Science course, BINUS University, 4th semester 2026.

| Member | Contribution |
|---|---|
| Veby Novalisa | Model implementation and experiment execution; Methodology and Results & Discussion sections; research poster (planning, content preparation, editing, production) |
| Haniah Humayra | Methodology and Results & Discussion sections; manuscript revision, language refinement, and formatting; research trailer |
| Evelyn Simin | Introduction section; presentation materials (slides and poster); conference presentation video |
| All members | Research topic selection and methodology discussion; Literature Review and Conclusion; final paper writing, review, and revision |

Instructors: Said Achmad, S.Kom., M.Kom; Maulin Nasari, S.T., M.Kom

## Background
Anemia (Hb < 11 g/dL) is typically detected through a Complete Blood Count (CBC) test, which is invasive, requires laboratory facilities and trained staff, and costs Rp100,000-300,000 per test. In areas with limited lab access, misdiagnosis rates can reach 50%. Nail bed pallor is a known non-invasive clinical indicator of anemia, but manual visual inspection only reaches 50-70% accuracy due to observer bias and lighting/skin-tone variation. This project addresses that gap with a deep learning pipeline that classifies anemia status directly from fingernail images.

## Scope

| Aspect | Detail |
|---|---|
| Task | Binary classification: anemic vs non-anemic |
| Input | Fingernail RGB images |
| Dataset | 2 combined public datasets, 4,466 images (2,564 anemic, 1,902 non-anemic) |
| Split | 70% train / 15% validation / 15% test (stratified, seed 42) |
| Architectures | DenseNet169, InceptionV3, Xception (transfer learning) |
| Framework | TensorFlow / Keras |

### About the dataset size
The 4,466 images come from two sources:

| Dataset | Anemic | Non-anemic | Total |
|---|---|---|---|
| Ghana (Mendeley Data) | 2,521 | 1,695 | 4,216 |
| Skin & fingernails (Figshare) | 43 | 207 | 250 |
| Combined | 2,564 | 1,902 | 4,466 |

According to the paper, the Ghana dataset consists of 710 original photos that were expanded to 4,216 images with augmentation by the dataset's authors. Because our split is done per image, near-duplicate copies of the same photo can end up in both the training and test sets, so the reported scores are probably optimistic (see [Limitations](#limitations)).

## Approach
The pipeline follows a sequential workflow from raw images to evaluated models:

1. **Data Collection**: Combined the Ghana/Mendeley fingernail dataset (pre-labeled) and the SpringerNature/Figshare fingernail + clinical metadata dataset (labeled from the Hb column, threshold 11 g/dL; g/L values are converted to g/dL).
2. **Image Processing**: Illumination normalization via CLAHE (LAB color space); nail area extraction (bounding box from metadata where available, YCbCr-based skin segmentation otherwise); resize to 224×224; pixel normalization to [0,1].
3. **Data Splitting**: Stratified 70/15/15 split (about 3,126 train, 670 validation, 670 test) to preserve class balance.
4. **Model Training**: Transfer learning on DenseNet169, InceptionV3, and Xception in two phases:
   - Phase 1: train a new classification head on a frozen pre-trained base (up to 20 epochs, Adam, lr 1e-4).
   - Phase 2: fine-tune the top 30 layers of the base (up to 50 epochs, Adam, lr 1e-5).
   - Batch size 16, online augmentation (horizontal flip, brightness 0.8-1.2, rotation up to ±20°), early stopping on validation loss, and the best weights kept by validation accuracy.
5. **Evaluation**: Accuracy, precision, recall, F1-score, and AUC-ROC on a held-out test set of 670 images (385 anemic, 285 non-anemic), with confusion matrices.

## Key Results

| Metric | DenseNet169 | InceptionV3 | Xception |
|---|---|---|---|
| Accuracy | 90.90% | **91.94%** | 80.45% |
| Precision | 90.90% | 91.94% | 80.48% |
| Recall | 90.90% | 91.94% | 80.45% |
| F1-Score | 90.90% | 91.92% | 80.23% |
| AUC-ROC | 0.9550 | **0.9703** | 0.8747 |
| Anemic precision | 92.19% | 92.11% | 80.24% |
| Anemic recall | 91.95% | **94.03%** | 87.53% |
| Non-anemic recall | 89.47% | 89.12% | 70.88% |

Precision, recall, and F1-score are weighted averages over both classes (which is why weighted recall equals accuracy). Anemic precision, anemic recall, and non-anemic recall are per class; anemic recall is computed from the confusion matrices.

**InceptionV3** achieved the best overall performance, with the highest recall on the anemic class (94.03%, only 23 false negatives), an important property for a screening tool, where a missed anemia case carries more risk than a false alarm. All three models were evaluated on the same held-out test set with identical preprocessing and training settings.

## Reproducibility notes (DenseNet169)
The DenseNet169 results need a short explanation, because they come from more than one run:

- **Reported metrics (90.90% accuracy, 0.9550 AUC-ROC):** from the main training run of `anemia_detection_training.py`. The plots for DenseNet169 were not produced by that run.
- **`evaluate_densenet.py`:** re-evaluates the saved best DenseNet169 weights (`output_results/DenseNet169_best.weights.h5`) on the same test split, **without retraining**. Preprocessing, split, and model definition are identical to the training script. It reproduced the reported metrics exactly and was used to regenerate the DenseNet169 confusion matrix.
- **`DenseNet169_history.png`:** comes from a separate DenseNet169 training run with the same code and settings. That run reached 90.0% test accuracy, so its curve does not belong to the 90.90% weights. Training is not fully deterministic (batch order, augmentation, dropout), so two runs of the same code give slightly different results.
- Each model was trained once on a single split, so there is no cross-validation or confidence interval.

## Limitations
- The split is per image. The Ghana dataset contains augmented copies of its original photos, so near-duplicates can land in both train and test, and the scores are probably optimistic.
- Classes are imbalanced, and the second dataset has only 250 images (43 anemic).
- The two datasets are preprocessed differently (YCbCr segmentation vs bounding boxes) and have very different class mixes, so a model could partly learn which dataset a photo came from.
- Binary labels are hard to judge near the 11 g/dL threshold.
- No Grad-CAM, no external dataset, and no testing in a real clinical setting.

## File Structure
```
anemia-detection/
├── anemia_detection_training.py    ← training & evaluation code for all 3 models
├── evaluate_densenet.py            ← re-evaluates saved DenseNet169 weights (no training)
├── paper.pdf                       ← the research paper
├── Poster.png                      ← course research poster
├── Outputs_.zip                    ← evaluation outputs
│   ├── dataset_distribution.png              ← combined dataset class distribution
│   ├── DenseNet169_confusion_matrix.png       ← confusion matrix (from evaluate_densenet.py)
│   ├── DenseNet169_history.png                ← accuracy & loss curves (separate run, see notes above)
│   ├── InceptionV3_confusion_matrix.png       ← confusion matrix, InceptionV3 test results
│   ├── InceptionV3_history.png                ← accuracy & loss curves, InceptionV3
│   ├── Xception_confusion_matrix.png          ← confusion matrix, Xception test results
│   └── Xception_history.png                   ← accuracy & loss curves, Xception
└── README.md
```

## Dataset
Available at the public sources below:

- [Detection of Anemia Using Colour of the Fingernails Image Datasets from Ghana](https://data.mendeley.com/datasets/2xx4j3kjg2/1) (Mendeley)
- [Dataset of Human Skin and Fingernails Images for Non-Invasive Haemoglobin Level Assessment](https://springernature.figshare.com/articles/dataset/Dataset_of_human_skin_and_fingernails_images_for_non-invasive_haemoglobin_level_assessment/25867432) (SpringerNature Figshare)

## Running the Code
1. Set up the two dataset folders in the project root:
   ```
   dataset1/
   ├── anemic/
   └── non_anemic/

   dataset2/
   ├── photo/
   └── metadata.csv
   ```
2. Install dependencies:
   ```
   pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn
   ```
3. Run training and evaluation of all three models:
   ```
   python anemia_detection_training.py
   ```
   All outputs (confusion matrices, training history, ROC curve, metrics comparison, summary report) are saved to `output_results/`, including the best weights of each model.
4. (Optional) Re-evaluate the saved DenseNet169 weights without retraining:
   ```
   python evaluate_densenet.py
   ```
   This needs `output_results/DenseNet169_best.weights.h5` from step 3, and saves its outputs (confusion matrix, ROC curve, metrics bar chart, summary) to `Densenet_result/`.

## Paper and Poster
- **Paper:** [`paper.pdf`](paper.pdf). The paper has been submitted to our supervising lecturers and is awaiting their decision on conference submission.
- **Poster:** [`Poster.png`](Poster.png), made for the Final Project / AOL assignment of the Research Methodology in Computer Science course.

## Dataset References
- J. W. Asare, P. Appiahene, and E. Donkoh, "Detection of Anemia Using Colour of the Fingernails Image Datasets from Ghana," Mendeley Data, 2022, doi: 10.17632/2xx4j3kjg2.1
- B. Yakimov et al., "Dataset of Human Skin and Fingernails Images for Non-Invasive Haemoglobin Level Assessment," Scientific Data, 2024, doi: 10.1038/s41597-024-03895-9
