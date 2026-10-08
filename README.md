# Bayesian Classification and Gaussian Discriminant Analysis (MATLAB)

Three experiments on Bayesian decision theory, written in MATLAB for a graduate machine-learning course (EECE 5644, Northeastern University, Fall 2025, Assignment 1).

| Question | Topic | Script |
|---|---|---|
| 1 | Binary classification on a 3-D Gaussian mixture: optimal ERM classifier, "naive Bayes" with identity covariances, and Fisher LDA, compared with ROC curves | `Question_1.m` |
| 2 | 4-class problem in 2-D: minimum-error (MAP) classifier versus an ERM classifier with a custom loss matrix | `Question_2.m` |
| 3 | Gaussian MAP classifiers on two real datasets (Wine Quality, Human Activity Recognition) with regularised covariances and PCA plots | `Question_3.m` |

The full write-up (derivations, figures, discussion) is in [`REPORT.pdf`](REPORT.pdf).

## Repository layout

```
Data_Generation.m   generates the Question 1 dataset and saves data.mat
Question_1.m        ERM / naive Bayes / Fisher LDA, ROC curves, P(error) vs threshold
Question_2.m        MAP vs custom-loss ERM on 4 Gaussian classes
Question_3.m        Wine Quality and HAR classification, PCA visualisation
evalGaussian.m      Gaussian pdf helper (from the course sample code)
Outputs/            figures and MATLAB command-window screenshots
REPORT.pdf          report
```

## Requirements

- MATLAB with the **Statistics and Machine Learning Toolbox** (`mvnrnd`, `randsrc`).
- For Question 3 only, download the datasets from the UCI Machine Learning Repository and place the files next to the scripts:
  - [Wine Quality](https://doi.org/10.24432/C56S3T): `winequality-white.csv` (semicolon-separated).
  - [Human Activity Recognition Using Smartphones](https://doi.org/10.24432/C51S4B): `X_train.txt`, `y_train.txt`, `X_test.txt`, `y_test.txt`.

## How to run

```matlab
Data_Generation   % creates data.mat (10,000 samples, priors 0.65 / 0.35)
Question_1        % ROC curves, P(error) vs threshold, results in the command window
Question_2        % classification plots and confusion matrices
Question_3        % requires the UCI files described above
```

## Methods in brief

- **Question 1:** samples are drawn from two 3-D Gaussians (means `[-0.5 -0.5 -0.5]` and `[1 1 1]`). The optimal classifier is a likelihood-ratio test; a threshold sweep (1,001 log-spaced values) produces the ROC curve and the minimum-error threshold, which is compared with the theoretical threshold `p0 / (1 - p0)`.
- **Question 2:** posteriors are computed for four Gaussian classes. MAP uses a 0-1 loss; the ERM variant uses the loss matrix `[0 1 1 3; 1 0 1 3; 1 1 0 3; 1 1 1 0]` so that missing class 4 costs three times as much.
- **Question 3:** class priors, means and covariances are estimated from data; each covariance is regularised with `lambda * I`, where `lambda = 0.01 * trace(Sigma) / rank(Sigma)`; results are shown as confusion matrices and PCA projections.

## Results reported in `REPORT.pdf`

| Experiment | Reported result |
|---|---|
| Q1 optimal ERM | P(error) 5.90 %, empirical threshold 1.91 vs theoretical 1.87 |
| Q1 naive Bayes (identity covariance) | 7.15 % |
| Q1 Fisher LDA | 6.63 % (see known issues) |
| Q2 MAP / custom-loss ERM | 17.89 % / 20.45 % (class-4 recall 30.6 % to 60.5 %) |
| Q3 | see known issues below |

Sample figures: `Outputs/Q1_ROC_Curves.png`, `Outputs/Q1_P_error_vs_Threshold.png`, `Outputs/Q2_PartA_MAP_Classification.png`, `Outputs/Q2_PartB_ERM_Classification.png`, `Outputs/Q3_Human_activity_classification.png`.

## Known issues

These are open problems found while auditing the repo; treat the numbers above with care until they are fixed.

- **Fisher LDA bug:** in `Question_1.m` the within-class scatter is computed as `Sigma_LDA(:,:,1) + Sigma_LDA(:,:,1)`; it should add the covariances of classes 1 and 2. The reported LDA error (6.63 %) comes from the buggy version.
- **"Naive Bayes" is simplified:** both class covariances are set to the identity matrix. A true naive Bayes model would use diagonal covariances estimated from the data.
- **Question 2 is not reproducible:** class means and covariances are random and no `rng` seed is set, so each run gives different results. The saved screenshots in `Outputs/` (for example 5.70 % and 5.84 % error) do not match the report (17.89 % and 20.45 %).
- **Question 3 numbers disagree:** the code reads the white-wine file, while comments and the report text say red wine. The command-window screenshot in `Outputs/Q_3_Results.png` (also embedded in the report) shows white wine, 4,898 samples, 69.33 % error, and a Human Activity Recognition error of 83.28 % with every sample assigned to the first class; the report text states 46.72 % and 7.29 %.
- `Outputs/Q_1_Results.png` shows the 4-class (Question 2) output rather than the Question 1 results.
- The Wine Quality and Human Activity Recognition data files are not included.

## Acknowledgements

Based on the structure of the sample code provided by the course instructor (`evalGaussian.m` and the ERM examples). The report states that a generative AI assistant (Google Gemini) was used for explanations, debugging and report drafting.

## License

MIT License (see `LICENSE`).
