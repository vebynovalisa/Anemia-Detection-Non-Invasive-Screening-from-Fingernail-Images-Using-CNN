# Anemia Detection from Fingernail RGB Images Using CNN
A research project applying a machine learning approach for non-invasive anemia screening using fingernail RGB images, built with transfer learning across three CNN architectures (DenseNet169, InceptionV3, Xception), trained and evaluated on a combined public dataset of 4,466 images.

## Team & Contributions
This project was completed collaboratively as a Final Project and Assurance of Learning (AOL) assignment for the Research Methodology in Computer Science course.

| Name | Student ID |
|---|---|
| Veby Novalisa | 2802552811 |
| Haniah Humayra | 2802564446 |
| Evelyn Simin | 2802405131 |

Instructors: Said Achmad, S.Kom., M.Kom (D6667); Maulin Nasari, S.T., M.Kom (D7082)
Course: Research Methodology in Computer Science - 4th Semester 2026, BINUS University

## Background
Anemia (Hb < 11 g/dL) is typically detected through a Complete Blood Count (CBC) test, which is invasive, requires laboratory facilities and trained staff, and costs Rp100,000-300,000 per test. In areas with limited lab access, misdiagnosis rates can reach 50%. Nail bed pallor is a known non-invasive clinical indicator of anemia, but manual visual inspection only reaches 50-70% accuracy due to observer bias and lighting/skin-tone variation. This project addresses that gap by building a deep learning pipeline that classifies anemia status directly from fingernail images.

## Scope

| Aspect | Detail |
|---|---|
| Task | Binary classification - anemic vs non-anemic |
| Input | Fingernail RGB images |
| Dataset | 2 combined public datasets, 4,466 images total (2,564 anemic, 1,902 non-anemic) |
| Split | 70% train / 15% validation / 15% test (stratified) |
| Architectures | DenseNet169, InceptionV3, Xception (transfer learning) |
| Framework | TensorFlow / Keras |

## Approach
The pipeline follows a sequential workflow from raw images to evaluated models:

1. **Data Collection** - Combined two public datasets: the Ghana/Mendeley fingernail dataset (pre-labeled) and the SpringerNature fingernail + clinical metadata dataset (labeled from the Hb column, threshold 11 g/dL).
2. **Image Processing** - Illumination normalization via CLAHE (LAB color space), nail area extraction (bounding box from metadata where available, YCbCr-based segmentation otherwise), resize to 224×224, pixel normalization to [0,1].
3. **Data Splitting** - Stratified 70/15/15 split to preserve class balance across train, validation, and test sets.
4. **Model Training** - Transfer learning on DenseNet169, InceptionV3, and Xception, with online augmentation (rotation, flip, brightness, noise), Adam optimizer (lr 1e-4), batch size 16, up to 50 epochs with early stopping, in two phases (feature extraction, then fine-tuning).
5. **Evaluation** - Accuracy, precision, recall, F1-score, and AUC-ROC on a held-out test set of 670 images, backed by confusion matrices and ROC curves.

## Key Results

| Metric | DenseNet169 | InceptionV3 | Xception |
|---|---|---|---|
| Accuracy | 90.90% | **91.94%** | 80.45% |
| Precision | 90.90% | 91.94% | 80.48% |
| Recall | 90.90% | 91.94% | 80.45% |
| F1-Score | 90.90% | 91.92% | 80.23% |
| AUC-ROC | 0.9550 | **0.9703** | 0.8747 |

**InceptionV3** achieved the best overall performance, with the highest recall on the anemic class (94.03%, only 23 false negatives) - an important property for a medical screening tool, where missed anemia cases carry more risk than false alarms.

Validation: All three models were evaluated on the same held-out test set (670 samples: 385 anemic, 285 non-anemic) using identical preprocessing and training settings.

## File Structure
```
anemia-detection/
├── anemia_detection_training.py    ← training & evaluation code for all 3 models
├── Poster.png                      ← course research poster
├── Outputs_.zip                    ← evaluation outputs
│   ├── dataset_distribution.png              ← combined dataset class distribution
│   ├── DenseNet169_confusion_matrix.png       ← confusion matrix, DenseNet169 test results
│   ├── DenseNet169_history.png                ← accuracy & loss curves, DenseNet169
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
3. Run training:
   ```
   python anemia_detection_training.py
   ```
4. All outputs (confusion matrices, training history, ROC curve, metrics comparison, summary report) are saved automatically to `output_results/`.

## Notes
This poster is a Final Project / Assurance of Learning (AOL) assignment for the Research Methodology in Computer Science course, separate from the paper, which has already been submitted to the supervising lecturers and is now awaiting their decision on conference submission. **The full paper is not included in this repo.**

## Dataset References
- J. W. Asare, P. Appiahene, and E. Donkoh, "Detection of Anemia Using Colour of the Fingernails Image Datasets from Ghana," Mendeley Data, 2022, doi: 10.17632/2xx4j3kjg2.1
- B. Yakimov et al., "Dataset of Human Skin and Fingernails Images for Non-Invasive Haemoglobin Level Assessment," Scientific Data, 2024, doi: 10.1038/s41597-024-03895-9
