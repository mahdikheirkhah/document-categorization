Here is the customized `CONTRIBUTING.md` tailored specifically for the Credit Scoring and Econometric Machine Learning project. I have updated the examples, libraries, architecture guidelines, and data integrity sections to reflect the nuances of predicting loan defaults, handling relational financial data, and building highly interpretable hybrid models.

---

# Contributing to the Credit Scoring & Econometric ML Project

Thank you for your interest in contributing to our credit scoring and risk assessment project! To ensure a smooth collaboration, maintain a high standard of code quality, and guarantee that our models meet both strict financial regulatory standards and high predictive accuracy (AUC), we require all contributors to strictly adhere to the following guidelines.

## 1. Development Workflow (Branching & CI/CD)

* **Branching Strategy:** Never commit or push directly to the `main` branch. Whether you are fixing a bug, adding a feature, or refactoring, you must create a dedicated branch (e.g., `feature/bureau-data-aggregation`, `fix/target-leakage`, `experiment/double-tree-architecture`).
* **CI/CD Checks:** Your branch must successfully pass all automated CI/CD pipelines before it can be merged. This includes formatting checks, linting, and automated tests.
* **Merging:** Once your pipeline passes, open a Pull Request (PR) to `main` and request a code review.

## 2. Dependency Management & Formatting

* **Poetry:** We use **Poetry** for package dependency resolving and environment management. Make sure you install dependencies using `poetry install` (ensure `loguru`, `pandas`, `scikit-learn`, `xgboost` or `lightgbm`, `shap`, and `dash` are added to your dependencies).
* **Black Formatter:** We enforce a uniform code style. Before committing your code, you must format your files using Black:

```bash
poetry run black .

```

## 3. Architecture & Paradigm

* **Object-Oriented Programming (OOP):** All code should be structured using OOP principles. Encapsulate related logic within well-defined classes. For example, structure your code into distinct classes for `RelationalFeatureEngineer` (handling Bureau/POS data), `DoubleTreeEstimator` (handling the decision-tree routing and logistic regression leaves), and `InterpretabilityExplainer` (handling global feature importance and local SHAP values).

## 4. Coding Standards & Naming Conventions

* **Variable Naming:** Use clear, descriptive variable names. Ensure your naming conventions follow standard Python Regex rules (e.g., `^[a-z_][a-z0-9_]*$` for snake_case variables and functions, `^[A-Z][a-zA-Z0-9]*$` for PascalCase classes). Always retain the `SK_ID_CURR` naming convention for client identifiers to avoid join errors.
* **Logging over Printing:** **Never use the standard `print()` statement.** We use **Loguru** for logging to eliminate boilerplate configuration and maintain clean code. Always use it to record pipeline flow, model evaluation metrics, and errors.

```python
from loguru import logger

logger.info("Successfully merged Bureau and POS_CASH balance data.")
logger.warning("Test set AUC dropped to 0.54; threshold of 0.55 not met. Check for overfitting.")
logger.error("Failed to compute SHAP values; model explainer expected different input dimensions.")

```

## 5. Function & Method Design

* **Single Responsibility Principle:** Make your functions and methods as reusable as possible. Each function/method should serve **exactly one purpose**. Break large, monolithic functions into smaller, modular pieces.
* **Type Hinting:** You must explicitly declare the data types for all arguments and the return type for every function and method.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_econometric_leaf(X_train: pd.DataFrame, y_train: pd.Series, max_iter: int = 100) -> LogisticRegression:

```

* **Documentation (Docstrings):** Every function and method must include a comprehensive docstring that clearly explains:
1. The goal and behavior of the function.
2. The types and descriptions of the input parameters.
3. The type and description of the output/return value.



## 6. Error Handling

* **Mandatory Try/Except:** Use `try` and `except` blocks thoroughly. You must include exception handling in **each function and method**.
* **Granular Handling:** Ensure that every distinct logic flow or block within your methods is wrapped in appropriate error handling to catch specific exceptions and log them accordingly using Loguru (e.g., `KeyError` for missing `SK_ID_CURR` columns, `ValueError` for passing raw data containing `NaNs` into the Logistic Regression model).

## 7. Model Integrity & Regulatory Compliance

* **Zero Data Leakage:** Ensure absolutely no future information or target variables leak into the training features. When aggregating historical data (like `previous_application.csv` or `installments_payments.csv`), explicitly verify that aggregations only look at past events relative to the current application.
* **Class Imbalance Awareness:** Because credit defaults are rare, traditional accuracy metrics are strictly forbidden for evaluation. All models must be evaluated using the Area Under the ROC Curve (AUC). Always use `StratifiedKFold` when splitting the data to preserve the default/non-default ratio.
* **Reproducibility:** Machine learning models, SHAP explainers, and cross-validation splits must be fully reproducible (this is a regulatory requirement for financial models). Always set and document `random_state` seeds for any stochastic processes (like standard Scalers, ML models, or splitters).

## 8. Testing

* **Test-Driven Collaboration:** If you write a new function or method, you are required to write the corresponding test code.
* **Flow Coverage:** Your tests must account for different logic flows and edge cases inside the method, including testing the `except` blocks by triggering known errors. Test edge cases specific to financial data, such as highly skewed income outliers, clients with absolutely no previous credit history, and extreme class imbalances.
* **Interpretability Checks:** Write tests to ensure that the sum of the baseline score and the SHAP feature contributions exactly equals the final predicted probability outputted by the model.