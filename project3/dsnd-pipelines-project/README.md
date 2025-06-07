# Data Science Pipeline for Fashion Forward Forecasting

## Project Summary
StyleSense, a fast-growing online women's fashion retailer, has experienced a surge in customer reviews—but many of these reviews are missing structured feedback on whether customers would recommend the product. Despite this, customers often leave rich text reviews that reflect their satisfaction.

My task was to use the existing, fully-labeled data to build a machine learning pipeline that **predicts whether a customer would recommend a product**. The model analyzes review text, customer age, product category, and other relevant information. By automating this process, I aimed to help StyleSense gain insights into customer satisfaction, identify trending products, and improve the shopping experience for its expanding customer base.

## Important Files
### Source Data 
Source data is located at `project3\dsnd-pipelines-project\starter\data\reviews.csv`.
### Notebook
The notebook for EDA and modeling is located at `project3\dsnd-pipelines-project\starter\starter.ipynb`.

## Notebook Structure

### Basic task
- load data
- prepare features (X) and target (y)

### My Work
- data exploration:
    - check missing values and found none
    - create a new categorical feature `Age_Group`
    - separate numerical, categorical and text features
    - numerical features: check main stats
    - categorical features:
      - check value distribution by target;
      - plot a line chart to show recommendation percentage by age group
- building pipeline
  - num_pipeline: using `MinMaxScaler()`
  - cat_pipeline: using both `OrdinalEncoder` and `OneHotEncoder`
  - text_pipeline: built custom transformers `ExtractPOSFeatures` and `ExtractEntityFeatures` using parts of speech (POS) tags and named entity recognition (NER), 
- training pipeline
  - using `RandomForestClassifier` as classification algorithm
  - evaluating with accuracy score and ROC AUC score
- fine-tuning pipeline: under `RandomizedSearchCV` over multiple max_features, n_estimators, and max_depth values with 3-fold CV implemented

## Built With

The main packages I used include:
- Numpy: https://github.com/numpy/numpy
- Pandas: https://github.com/pandas-dev/pandas
- Matplotlib: https://github.com/matplotlib/matplotlib
- Seaborn: https://github.com/mwaskom/seaborn
- Sklearn: https://github.com/scikit-learn/scikit-learn
- Spacy: https://github.com/explosion/spaCy
