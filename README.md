# Sentiment Analysis of Financial News Headlines

This project builds a simple sentiment analysis model for financial news headlines about gold prices. The primary goal is to classify each headline into one of the following **price sentiment** classes while also gaining hands-on experience with the statistical and mathematical ideas behind text classification:

- `positive` – headline suggests upward movement in price
- `negative` – headline suggests downward movement in price
- `neutral` – headline suggests sideways or steady movement in price
- `none` – no clear assessment about price movement (these rows are dropped for modeling)

In addition to building a working model, the project is designed to:

- Practice key **statistics and machine learning concepts** such as train/test splitting, out-of-sample evaluation, and confusion-matrix analysis.
- Understand the **underlying math** behind TF–IDF vectorization and linear Support Vector Machines as maximum-margin classifiers in high-dimensional feature spaces.
- Provide a compact, implementation-focused example suitable for a **statistics-oriented MS application portfolio**.

The core implementation is contained in the Jupyter notebook:

- `Sentiment_main.ipynb`

## Dataset

The notebook expects a CSV file containing news headlines and their labeled price sentiment.

- Expected file path (as used in the notebook):
  - `data/gold_headlines.csv`
- Expected columns:
  - `News` – the text of the news headline
  - `Price Sentiment` – label for price movement (`positive`, `negative`, `neutral`, or `none`)

Rows with `Price Sentiment == 'none'` are excluded from training and evaluation.

> Note: Update the CSV path in the notebook if your dataset is stored in a different location.

## Environment and Dependencies

This project uses Python and standard data science / machine learning packages.

Main libraries used:

- `numpy`
- `pandas`
- `scikit-learn`
  - `TfidfVectorizer` from `sklearn.feature_extraction.text`
  - `train_test_split` from `sklearn.model_selection`
  - `accuracy_score`, `f1_score`, `recall_score`, `precision_score`, `confusion_matrix` from `sklearn.metrics`
  - `SVC` from `sklearn.svm`
- `re` (Python regular expressions)

### Example installation

Use a virtual environment and install the required packages (adjust as needed):

```bash
pip install numpy pandas scikit-learn
```

## Notebook Overview

The notebook `Sentiment_main.ipynb` performs the following steps:

1. **Load packages**
   - Imports all required Python libraries.

2. **Load dataset**
   - Reads the CSV file:
     ```python
     df = pd.read_csv("data/gold_headlines.csv")
     ```

3. **Filter labels**
   - Drops rows where `Price Sentiment` is `'none'` so that only informative labels remain.

4. **Exploratory display**
   - Prints and displays the `News` and `Price Sentiment` columns for inspection.

5. **Text cleaning**
   - Defines a `cleaner` function to preprocess headlines:
     - Removes `@mentions`
     - Removes URLs beginning with `http`
     - Removes special characters and punctuation (while leaving emojis intact)

6. **TF–IDF + SVM model (statistical view)**
   - Defines `headline_sentiment(df)` which:
     - Extracts `News` and `Price Sentiment` columns
     - Cleans headlines using the `cleaner` function
     - Splits data into train and test sets (80/20) using `train_test_split` (as specified by `test_size = 0.2`)
     - Vectorizes text using `TfidfVectorizer(sublinear_tf=True, use_idf=True)`
       - Each headline is represented as a high-dimensional feature vector of term TF–IDF scores (term frequency reweighted by inverse document frequency).
     - Trains a linear-kernel SVM classifier:
       ```python
       SVM_L = SVC(kernel='linear')
       SVM_L.fit(train_corpus_tf_idf, Y_train)
       ```
       - Mathematically, the linear SVM learns one or more separating hyperplanes in the TF–IDF feature space that maximize the margin between sentiment classes.
     - Predicts on the test set
     - Prints test accuracy (overall proportion of correctly classified headlines)
     - Builds and displays a confusion matrix using `pandas.DataFrame`, which shows per-class performance and misclassification patterns.
   - The function returns the fitted `TfidfVectorizer` and trained `SVC` model:
     ```python
     vectorizer, model = headline_sentiment(df)
     ```

7. **Sample predictions**
   - Uses the trained `vectorizer` and `model` to classify example headlines, for instance:
     ```python
     vector = vectorizer.transform(["The price of gold continues declining."])
     sentiment = model.predict(vector)
     print(sentiment)
     ```

8. **Notes on performance (statistical interpretation)**
   - The confusion matrix typically shows that performance on the `neutral` class is weaker, while `positive` and `negative` are more reliably identified.
   - This allows a more nuanced, statistically grounded discussion of model quality than accuracy alone (e.g., asymmetry between false positives and false negatives for different classes).

9. **Future improvements**
   - The notebook suggests trying **sentence transformers** to significantly improve results beyond the simple TF–IDF + SVM baseline.

## How to Run the Notebook

1. Ensure Python and the required libraries are installed.
2. Place the dataset CSV at the path used in the notebook (`data/gold_headlines.csv`) or adjust the path in the `pd.read_csv(...)` call.
3. Open `Sentiment_main.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
4. Run all cells in order:
   - Load packages
   - Load and filter data
   - Define cleaning and modeling functions
   - Train the model and inspect accuracy and confusion matrix
   - Optionally, run the sample prediction cells with your own headlines

## Outputs

### Console / notebook output

- Overall test accuracy
- Confusion matrix for the test set
- Sample predicted sentiments for custom headlines

### Generated files

- `predicted.csv` – saved in the current working directory when `headline_sentiment(df)` is executed, containing:
  - `News`
  - `Actual Price Sentiment`
  - `Predicted Sentiment`

No other files are written by the notebook in its current form.

## Extending the Project

Some possible extensions:

- Replace TF–IDF with pre-trained sentence embeddings (e.g., sentence transformers) and retrain the classifier.
- Perform hyperparameter tuning of the SVM (e.g., C, class weights).
- Add additional evaluation metrics (macro/micro F1, per-class precision/recall) to deepen the statistical analysis.
- Experiment with alternative models such as logistic regression, random forests, or gradient boosting.
- Apply the same workflow to other financial instruments or asset classes by changing the dataset.
