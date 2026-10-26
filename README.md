
#  Predict Late Deliveries
A food delivery company wants to know, at the moment an order is placed, whether it will arrive late, so that 
it can offer a small voucher before the customer is disappointed. 

## What am I trying to save?

A $4 voucher is sent to every order we flag, even when that order
arrives on time. If we miss a late order, the loss is $15.

Cost = $4 × number flagged + $15 × number late but NOT flagged.

Flagging nobody is the simple starting policy. My goal is to beat that
policy with data the model has never seen.

## 1. Set it up
conda env create -f environment.yml
conda activate module05-quality
python -m pip install -e .

python -m pip install -r requirements.txt
python -m pip install -e .


## 2. Run everything
python -m latedeliveries.train
python -m pytest
python -m ruff format --check .
python -m ruff check .

The train command starts again from the same provided generator,
compares models, searches settings, picks the voucher cutoff, then scores
the saved test set just once per run. The final outputs are in
`models/` and `reports/` (and they are kept in Git).

`ruff` uses rules E, F, I, B, UP in `pyproject.toml`. The generated
notebook is excluded from Ruff checks; package and test source are not.

## 3. The data and the rules

I added the `data.py` with its exact data-making logic in
its own commit. It has the supplied `make_orders(n=3000, seed=7)` function,
creates the same 3,000 rows, and sometimes leaves preparation time and
ratings empty.

### Which columns am I allowed to use?

Here is every data column:

| Column | Use? | My easy explanation |
| --- | --- | --- |
| `order_id` | No | Just a label, not a reason for delay. |
| `distance_km` | Yes | We know the restaurant and customer's location. |
| `items` | Yes | The basket's item count is already known. |
| `hour` | Yes | We know the hour when the customer checks out. |
| `weather` | Yes | We can check the weather right now. |
| `day_type` | Yes | We know if today is a weekday or weekend. |
| `vehicle` | No | A courier may not be picked at order time. |
| `prep_minutes` | No | Real kitchen preparation time comes later. |
| `courier_months` | No | We may not know which courier will deliver. |
| `customer_rating` | No | The customer gives a rating after delivery. |
| `late` | Target | This is what we are trying to guess. |

The five input columns are in `pipeline.py` only once:

- Numbers: `distance_km`, `items`, `hour`
- Categories: `weather`, `day_type`

I split with a single fixed seed, 7, and use a stratified 75/25
split: 2,250 training orders and 750 hidden test orders. Both have a
late rate of 15.6%. Stratified just means the two sets have about the
same amount of late orders.

## 4. The safe pipeline

My `build_pipeline(model)` is like an assembly line:

1. Fill missing number values using the training median.
2. Scale the numbers so they are easier to compare.
3. Turn words like `rain` into one-hot columns; ignore a new word like `hail`.
4. Let the chosen model predict a chance of late delivery.

Every candidate, search and final prediction uses that whole pipeline.
Preprocessing happens inside each training fold. This is important:
information from one fold cannot leak into the other fold. The pipeline
can also predict when `prep_minutes` is missing because it never uses
that future-only column in the first place.

## 5. Try models fairly (training data only)

I used the same five stratified folds for one DummyClassifier baseline
and four real models. Balanced logistic uses
`class_weight="balanced"`. The baseline is useful for comparison but cannot
be selected as the real model.

For each metric, the number before ± is the five-fold mean and
the number after it is the standard deviation (how much the folds differ).
The cost column uses the simple break-even threshold of $4 / $15 ≈ 0.267.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Cost / order |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.858 ± 0.006 | 0.644 ± 0.067 | 0.199 ± 0.019 | 0.304 ± 0.029 | 0.773 ± 0.035 | 1.891 ± 0.137 |
| Random forest | 0.857 ± 0.012 | 0.643 ± 0.131 | 0.191 ± 0.040 | 0.294 ± 0.062 | 0.754 ± 0.032 | 1.931 ± 0.130 |
| Decision tree | 0.855 ± 0.006 | 0.614 ± 0.067 | 0.193 ± 0.056 | 0.289 ± 0.066 | 0.741 ± 0.041 | 1.940 ± 0.139 |
| Flag-nobody baseline | 0.844 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | 2.340 ± 0.013 |
| Balanced logistic | 0.732 ± 0.018 | 0.325 ± 0.026 | 0.661 ± 0.044 | 0.436 ± 0.033 | 0.772 ± 0.035 | 3.029 ± 0.075 |

The full-precision numbers are saved in `reports/cv_results.csv`.

I chose logistic regression because its predicted voucher cost was
about $1.891 per training validation order, the smallest of the real
models. The dummy cost about $2.340 per order. Balanced logistic has
high recall, but it flags too many orders and spends more voucher money.
This is why I did not just choose the model with the best accuracy,
recall or ROC-AUC.

## 6. Tune just that model (still training only)
I used `GridSearchCV` on the full pipeline with five stratified folds.
It tries two settings and six combinations:

- `model__C`: `0.1`, `1.0`, `10.0`
- `model__class_weight`: `None`, `balanced`

I used ROC-AUC to find good order rankings instead of chasing easy
accuracy on the uncommon late class. Best settings:
`model__C = 0.1`, `model__class_weight = None`.

The best searched training ROC-AUC was 0.772862, versus 0.772782
for the untuned model, a tiny gain of 0.000080. The original
fold-to-fold standard deviation was 0.035244, much larger than that
gain. So NO, the gain was not bigger than the usual fold spread.
This comparison is just a guide, not proof that tuning really helped.

## 7. Pick the voucher threshold without peeking
Using `cross_val_predict`, I got each training row's chance from a model
that did not learn from that row (out-of-fold prediction). I tried
all 19 thresholds: 0.05 to 0.95, moving 0.05 each time, and used the
real QuickBite cost formula. If two thresholds tie, the smaller one wins.

My chosen threshold is 0.25. This means flag an order when its
predicted late chance is 25% or higher. The lowest training OOF cost
was $4,196 for the 2,250 training orders.
The chart and all 19 choices are saved in `reports/threshold_costs.csv`
and shown in the notebook. The test set did not choose this threshold.

To test the math by hand: if actual late flags are `[1, 0, 1, 0]`, the
predicted chances are `[.90, .55, .30, .10]`, and cutoff is `.15`,
we flag three orders and miss none: 3 × $4 = $12. Flagging nobody
would cost $30, and flagging everyone would cost $16.

## 8. The ONE held-out test result

Only after all model/parameter/threshold choices did I fit the final
pipeline to all 2,250 training orders and predict on 750 unseen test
orders. There were 117 actual late orders in the test set.

Here is the confusion matrix (rows = what really happened;
columns = what the model predicted):
| Actual / Predicted | On time | Late |
| --- | ---: | ---: |
| On time | 570 (true negative) | 63 (false positive) |
| Late | 70 (false negative) | 47 (true positive) |

The test numbers at threshold 0.25 are:
| Metric | Value |
| --- | ---: |
| Accuracy | 0.823 |
| Precision | 0.427 |
| Recall | 0.402 |
| F1 | 0.414 |
| ROC-AUC (continuous chances) | 0.752 |

### What would QuickBite pay for those 750 orders?
| Policy | Test cost |
| --- | ---: |
| Flag nobody | $1,755 |
| Flag everybody | $3,000 |
| Our chosen model | $1,490 |

So the model saved $265, or about
15.1%, compared with flagging nobody.
It is not perfect: 70 late orders were still missed. The experiment
is on invented orders, not a real customer trial.

The complete results are saved in `reports/metrics.json`.

## 9. Keep the trained model safe and reusable
`models/late_orders.joblib` contains the whole trained pipeline:
cleaning, one-hot columns and the classifier. A second program can load
it without repeating any feature processing:

```python
import joblib
import pandas as pd

model = joblib.load("models/late_orders.joblib")
new_order = pd.DataFrame([{
    "distance_km": 4.2,
    "items": 2,
    "hour": 18,
    "weather": "rain",
    "day_type": "weekday",
}])
chance = model.predict_proba(new_order)[0, 1]
print("Chance of lateness:", round(chance, 3))
print("Send voucher?", chance >= 0.25)
```

Only load a `.joblib` file you trust. It can run Python code during
loading, just like other pickle-based files. The matching
`models/model_info.json` contains the scikit-learn version, seed,
threshold, chosen settings, allowed/excluded columns and test results.

## 10. My notebook and tests
Open `notebooks/analysis.ipynb`. It is saved with outputs and shows:
training late rate, the full CV comparison, cost by training threshold,
the saved test confusion matrix and all three test costs. It ends
with a recommendation under 200 words. It only reads saved test
results instead of using those rows to make more choices.

The `tests/` folder has 28 passing pytest cases. They check the split,
feature exclusions, missing values, unknown categories, probabilities,
a pencil-and-paper voucher example, threshold ties, six search choices,
saved-file consistency and reloading a joblib model via `tmp_path`.
At least one test uses `@pytest.mark.parametrize`.

## 11. Folder layout
Humber-AIGC5005_Lab05/
├── .git/                     
├── .gitignore
├── README.md
├── environment.yml
├── requirements.txt
├── pyproject.toml
├── src/latedeliveries/
│   ├── __init__.py
│   ├── data.py               
│   ├── pipeline.py
│   ├── evaluation.py
│   └── train.py
├── tests/
│   ├── conftest.py
│   ├── test_pipeline.py
│   ├── test_evaluation.py
│   ├── test_tuning.py
│   ├── test_data.py
│   ├── test_saved_model.py
│   └── test_artifacts.py
├── notebooks/analysis.ipynb
├── models/late_orders.joblib
├── models/model_info.json
├── reports/cv_results.csv
├── reports/threshold_costs.csv
└── reports/metrics.json


## Would I use it?
Maybe, in a small pilot first. It saved money on the hidden test set,
but fake data can be different from real deliveries. I would collect
real-world orders, check what is truly known at checkout, and check that
voucher spending still goes down before giving it to actual customers.