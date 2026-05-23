#  KRONOS - Gym Retention

AI-powered gym member cancellation prediction system built with Streamlit and Scikit-learn.



##  Install Requirements

Install all required libraries:

```bash
pip install -r requirements.txt
```

### requirements.txt

```txt
streamlit
pandas
numpy
scikit-learn
joblib
plotly
```



##  Train the Models

Before running the app, train the machine learning models:

```bash
python train_model.py
```

This generates the following files:

* `cancellation_model.pkl`
* `cancellation_logistic_model.pkl`
* `cancellation_random_forest_model.pkl`
* `cancellation_gradient_boosting_model.pkl`



##  Run the App

Start the Streamlit dashboard:

```bash
streamlit run app.py
```



##  Required CSV Columns

Input CSV files must contain:

* `age`
* `months_since_joined`
* `join_month`
* `visits_per_week`
* `payment_delay_days`

Optional:

* `member_id`



##  Team

IART Project 2 — T09G07

* Catarina Guimarães
* Sara García
* Stavros Piperakis
