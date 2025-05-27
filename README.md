# Titanic: Passenger Survival Prediction

**Author**: Haowen Zheng

---

## Abstract

This project investigates the problem of predicting passenger survival on the Titanic using machine learning. The workflow covers data preprocessing, feature engineering, model training, evaluation, and test set submission. Four models were developed and compared: **Logistic Regression**, **Random Forest**, **XGBoost**, and a **Stacking Ensemble**.

During training and validation, ensemble models—especially XGBoost—demonstrated superior performance across multiple metrics such as AUC, average precision, and KS statistic. However, final Kaggle leaderboard results revealed that the simpler Logistic Regression model achieved the highest test accuracy (**0.78947**), outperforming the more complex models.

This unexpected outcome emphasizes the importance of model simplicity, proper regularization, and generalization over raw complexity. Key findings include the significance of features like **sex**, **ticket class**, **title**, and **cabin availability** in survival prediction, aligning with historical and social context[1][2][3].

**Next steps** may include testing advanced ensembling strategies, incorporating domain-specific feature engineering, or applying similar techniques to other imbalanced or high-stakes binary classification tasks.

---

## Rationale

**It builds practical machine learning skills.**  
The Titanic survival prediction problem is a classic example used to learn data science. It covers key steps like data preprocessing, feature engineering, model selection, and evaluation[7][8].

**It reflects real-world decision-making.**  
Although based on a historical event, the question of “Who survives?” relates closely to modern issues—like predicting outcomes in healthcare, disaster response, and safety systems.

**It helps understand the impact of human and social factors.**  
Variables such as gender, age, and ticket class reveal how social status and group identity influenced survival. This can lead to deeper thinking about fairness and ethics in machine learning[8][9].

**It trains us to balance simplicity and performance.**  
The project shows that simpler models (like logistic regression) can sometimes outperform complex ones. This highlights the importance of interpretability, robustness, and generalization.

> In short, this question offers a hands-on, meaningful way to explore both the technical and human sides of machine learning.

---

## Research Question

The central research question of this project is:

> **Can we accurately predict whether a passenger survived the Titanic disaster using information such as gender, age, passenger class, and other available features?**

To explore this question, the project aims to:

- Identify which features (e.g., sex, ticket class, family size, cabin information) are most predictive of survival.
- Build, tune, and compare several classification models to determine which performs best in terms of both training metrics and generalization to unseen data.
- Understand how different modeling approaches—ranging from simple linear classifiers to complex ensemble techniques—balance interpretability and predictive performance.
- Examine the generalization ability of models using real-world unseen test data, such as Kaggle's held-out leaderboard dataset.

Ultimately, the goal is to determine not only which model predicts survival most accurately, but also how well those predictions align with historical, social, and operational realities.


---

## Data Sources

This project uses the Titanic dataset provided by Kaggle as part of the Titanic: Machine Learning from Disaster competition. The dataset consists of two CSV files[1]:

- `train.csv`: Contains 891 labeled samples with both feature values and survival outcomes (`Survived` column).
- `test.csv`: Contains 418 unlabeled samples used for final model evaluation and leaderboard submission.

Each passenger entry includes a variety of features, such as:

- **Demographic information**: `Sex`, `Age`
- **Socioeconomic indicators**: `Pclass` (ticket class), `Fare`, `Cabin`
- **Relational context**: `SibSp` (siblings/spouses aboard), `Parch` (parents/children aboard)
- **Identifiers and metadata**: `Name`, `Ticket`, `PassengerId`

Additional features such as `Title` (extracted from `Name`) and `Has_Cabin` (binary indicator for cabin availability) were engineered during preprocessing.

To support the full data mining workflow, several intermediate and output files were generated during the project:

- `train_processed.xls` / `test_processed.xls`: Preprocessed datasets used for model training and testing. These include cleaned, encoded, and feature-engineered data.
- `logistic_regression_predictions.xls`, `random_forest_predictions.xls`, `xgboost_predictions.xls`, `stacking_ensemble_predictions.xls`: Contain predicted probabilities and/or binary classification results on the test set, output by each respective model.
- `logistic_regression_submission.xls`, `random_forest_submission.xls`, `xgboost_submission.xls`, `stacking_ensemble_submission.xls`: Kaggle-formatted CSV submission files (with `PassengerId` and `Survived` columns) for leaderboard evaluation.

These data sources enabled a complete supervised learning workflow—from feature extraction and model training to public test set evaluation through Kaggle’s submission system.


---

## Methodology

To address the binary classification task of predicting survival, this project followed a structured machine learning pipeline:

1. **Data Preprocessing**  
   The dataset underwent several cleaning and transformation steps:
   - **Missing values**:  
     - `Age`: filled using the median grouped by title (e.g., Mr., Mrs.).
     - `Embarked`: imputed with the most frequent value.
     - `Fare`: filled with the median based on class and embarkation port.
   - **Categorical Encoding**:  
     - Features like `Sex`, `Embarked`, and `Pclass` were encoded using one-hot encoding to enable use in tree-based and linear models.
   - **Feature Engineering**:  
     New features were created to enrich the predictive information:
     - `Title`: extracted from the `Name` column to represent social or family status (e.g., Mr, Miss, Master).
     - `Has_Cabin`: a binary feature indicating whether the `Cabin` field was present or missing.
     - `CabinLetter`: the first character of the `Cabin` string, used to approximate deck location.
     - `Family_Size`: calculated as the sum of `SibSp` and `Parch` plus one (the individual), representing the total number of family members aboard.

2. **Feature Standardization**  
   For algorithms sensitive to input scale (e.g., Logistic Regression), continuous variables were standardized using `StandardScaler` to ensure faster convergence and balanced coefficient estimation.

3. **Model Development**  
   The project implemented four classification models:
   - **Logistic Regression**: interpretable and robust linear classifier.
   - **Random Forest**: an ensemble of decision trees using bagging for variance reduction[4].
   - **XGBoost**: a gradient-boosted decision tree model known for high performance[5].
   - **Stacking Ensemble**: a meta-model (Logistic Regression) trained on the probabilistic outputs of the three base learners[6].

4. **Model Tuning**  
   Each model underwent hyperparameter optimization using `GridSearchCV` with 5-fold cross-validation, optimizing for validation accuracy.

5. **Performance Evaluation**  
   Models were evaluated with a consistent set of metrics to ensure fair and comprehensive comparison:
   - **Accuracy**: the proportion of correct predictions among all instances.
   - **Precision**: the proportion of predicted positives that are actually positive (i.e., correctness of positive predictions).
   - **Recall**: the proportion of actual positives that are correctly identified (i.e., model’s ability to detect positives).
   - **F1-Score**: the harmonic mean of precision and recall, balancing both metrics.
   - **ROC AUC**: the area under the Receiver Operating Characteristic curve, reflecting the model’s ability to distinguish between classes across all thresholds.
   - **KS Statistic**: the maximum distance between cumulative distributions of true positive and true negative rates.
   - **Average Precision (AP)**: the area under the precision-recall curve, emphasizing performance on imbalanced data.
   - **Threshold Analysis**: evaluated model behavior at different classification thresholds to identify the optimal trade-off between precision, recall, and F1-score.

6. **Final Testing and Submission**  
   The trained models were applied to the processed test dataset (418 passengers). Binary predictions were generated and submitted to the Kaggle competition for final benchmarking. The leaderboard scores were used as the final reference for each model’s generalization performance.


---

## Results

Through a systematic implementation and evaluation of four supervised classification models — **Logistic Regression**, **Random Forest**, **XGBoost**, and a **Stacking Ensemble** — the project yielded several key findings:

- **Logistic Regression** achieved the highest Kaggle test accuracy (**0.78947**), outperforming more complex models. Its simplicity and regularization contributed to superior generalization.

- **XGBoost** showed the best training and validation performance in terms of metrics such as **ROC AUC**, **KS Statistic**, and **Average Precision**, indicating strong learning capacity and ability to model complex feature interactions.

- **Random Forest** performed robustly across both validation and test sets, achieving an accuracy of **0.77511** on the test set. It offered a balance between model complexity and interpretability.

- The **Stacking Ensemble**, while theoretically designed to integrate the strengths of all models, slightly underperformed on the test set (**0.77033**), possibly due to overfitting or increased variance introduced by the stacking process.

In terms of feature importance, the models consistently highlighted the following variables as highly predictive:

- **Sex**: Female passengers had significantly higher survival rates.
- **Pclass**: Passengers in first class had better chances of survival.
- **Title**: Derived from the `Name` field, reflecting social status.
- **Family_Size**: Larger families were associated with lower survival.
- **Has_Cabin**: Availability of cabin information (proxy for wealth/status) had a positive correlation with survival.

These findings are consistent with historical accounts and provide both predictive power and human-understandable insights. The results reinforce the notion that **carefully engineered simple models can outperform complex ones** in settings with limited, noisy, or imbalanced data.

In terms of survival likelihood, the analysis revealed that **gender, passenger class, social title, and cabin information** were the most influential factors. Specifically:

- **Female passengers** had a significantly higher chance of survival, reflecting the "women and children first" evacuation protocol.
- **First-class passengers** were prioritized during rescue, possibly due to their proximity to lifeboats and crew.
- **Social title**, extracted from names (e.g., "Mrs", "Mr", "Miss"), served as a proxy for societal role and status, influencing both perceived importance and group behavior during the disaster.
- **Cabin availability** indicated higher ticket prices and status, correlating positively with survival, likely due to better location and earlier access to lifeboats.

These relationships not only improved predictive performance but also aligned with the historical and ethical realities of the Titanic tragedy.



---

## Next Steps

While the project achieved strong results and valuable insights, several opportunities for further improvement remain:

- **Advanced Ensembling**: Future work could experiment with more sophisticated ensemble strategies, such as blending or weighted averaging, to better combine model strengths[6][7].

- **Feature Enrichment**: Incorporating more domain-informed features—such as deck location from cabin numbers, family surnames, or ticket grouping—may improve the model’s ability to capture social dynamics.

- **Class Imbalance Handling**: Methods like SMOTE (Synthetic Minority Over-sampling Technique) or adjusting class weights could help boost recall on the minority (survived) class[7][8].

- **Model Interpretability**: Tools like SHAP or LIME could be integrated to further visualize and explain model decisions, especially for complex models like XGBoost[9][10].

- **Broader Validation**: Employing stratified or repeated K-fold cross-validation may give more stable and generalizable estimates of model performance[7].

- **Deployment Simulation**: Testing the model in a simulated application (e.g., evacuation prioritization scenario) could demonstrate its real-world utility and limitations.

By implementing these steps, future iterations of the project could enhance predictive accuracy, interpretability, and practical applicability across domains beyond maritime disaster prediction.


---

## Conclusion

This project successfully explored the Titanic passenger survival prediction task using a complete machine learning pipeline. Four models—Logistic Regression, Random Forest, XGBoost, and a Stacking Ensemble—were developed, optimized, and evaluated through cross-validation and Kaggle test submission.

### Positive Outcomes
- **Strong baseline performance**: Logistic Regression achieved the highest Kaggle test accuracy (0.78947), demonstrating the effectiveness of simple, interpretable models.
- **Effective feature engineering**: Variables such as `Sex`, `Pclass`, `Title`, and `Cabin availability` emerged as highly predictive, aligning with historical accounts of evacuation priorities and social hierarchies.
- **Robust evaluation framework**: A comprehensive set of evaluation metrics (AUC, F1-score, KS statistic, average precision) and threshold analysis helped assess model quality from multiple perspectives.

### Insights on Survival Rate
- **Gender**: Female passengers had significantly higher survival rates, consistent with the “women and children first” policy during evacuation.
- **Class and Status**: Passengers in 1st class or those with higher social titles (e.g., “Mrs.”, “Miss.”) were more likely to survive, indicating the influence of social and economic status.
- **Family and Cabin Info**: Smaller family sizes (low `SibSp` and `Parch`) and the presence of cabin information were associated with higher survival probabilities, possibly due to better access to escape routes or documentation priority.

These findings support the idea that survival on the Titanic was not random, but rather heavily influenced by **demographic, social, and logistical factors**.

### Limitations and Cautions
- **Generalization challenges**: More complex models like XGBoost and the stacking ensemble showed signs of slight overfitting, performing worse than expected on the unseen test set.
- **Data constraints**: The dataset size was modest (891 training samples), which limited the learning capacity of high-complexity models and the ability to explore deep feature interactions.
- **Noisy or missing data**: Imputation and one-hot encoding may have introduced bias, and certain features (like `Cabin`) had significant missingness.

### Final Recommendation
When working with limited, imbalanced, or noisy data, **simplicity and regularization can often outperform raw model complexity**. Logistic Regression, when properly tuned, provides a strong, explainable baseline and should be the first model tested in similar binary classification problems. However, ensemble models remain valuable for capturing non-linear patterns and should not be dismissed, especially when more data is available.

In conclusion, this project illustrates how thoughtful preprocessing, methodical evaluation, and a balance between complexity and interpretability lead to meaningful insights and strong real-world performance.


---

## Bibliography

[1] Kaggle. Titanic: Machine Learning from Disaster. Retrieved from https://www.kaggle.com/competitions/titanic

[2] Ai, Y. (2023). Predicting Titanic Survivors by Using Machine Learning. Highlights in Science, Engineering and Technology, 34, 360–366. https://www.researchgate.net/publication/369471638

[3] Zhang, Y. (2024). Titanic Survival Prediction Based on Machine Learning Algorithms. Highlights in Science, Engineering and Technology, 107, 189–195. https://doi.org/10.54097/mwkr1a24

[4] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

[5] Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

[6] Wolpert, D. H. (1992). Stacked Generalization. Neural Networks, 5(2), 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1

[7] Kuhn, M., & Johnson, K. (2013). Applied Predictive Modeling. Springer. https://doi.org/10.1007/978-1-4614-6849-3

[8] Provost, F., & Fawcett, T. (2013). Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking. O'Reilly Media.

[9] Lipton, Z. C. (2018). The Mythos of Model Interpretability. Queue, 16(3), 31–57. https://doi.org/10.1145/3236386.3241340

[10] Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS) (pp. 4765–4774). https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html


---

## Project Directory Structure

This repository is organized as follows:

```
├── data/                          # All data-related resources
│   ├── raw_data/                  # Original Titanic datasets (train.xls, test.xls)
│   ├── processed_data/            # Reprocessed datasets (train_processed.xls, test_processed.xls)
│   ├── predictions/               # Model prediction outputs (logistic_regression_predictions.xls, etc.)
│   └── submissions/               # Kaggle submission files (logistic_regression_submission.xls, etc.)

├── images/                        # All visualizations used in the project report
│   ├── part1/                     # Visualizations for Part 1 (e.g., EDA)
│   ├── part2/                     # Visualizations for Part 2 (e.g., feature engineering)
│   ├── part3/                     # Visualizations for Part 3 (e.g., Logistic Regression)
│   ├── part4/                     # Visualizations for Part 4 (e.g., Random Forest)
│   ├── part5/                     # Visualizations for Part 5 (e.g., XGBoost)
│   ├── part6/                     # Visualizations for Part 6 (e.g., Stacking Ensemble)
│   └── part7/                     # Visualizations for Part 7 (e.g., model comparison summary)

├── Datamining_project.ipynb       # Main Jupyter Notebook containing the modeling process
└── README.md                      # This file (project summary and structure)
```
Due to its large size (38MB), the project PPT file was submitted separately and is not included in this repository. 
---

#### Contact and Further Information
For questions, feedback, or further collaboration, please contact:

**Haowen Zheng**  
Email: zhenghaowen0526@hotmail.com  
GitHub: Kong6480
Project Repository: https://github.com/Kong6480/Datamining-Project-Titanic-Survival-Prediction

---



