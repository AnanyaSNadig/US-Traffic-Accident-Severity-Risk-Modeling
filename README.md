# US Traffic Accident Severity & Risk Modeling

A big-data-driven analysis of U.S. traffic accident records (2016–2023) using Apache Spark and PySpark. The project builds a scalable analytics pipeline to identify patterns associated with high-severity accidents and predicts severity using machine learning models.

**Course:** CSGY-6513 Big Data | Section C | Spring 2026

**Instructor:** Prof. Amit Patel

**Team:** 
- Ananya Nadig (an4968)
- Isha Jariwala (ij2221)
- Rhea Shastri (rs9459)

## Dataset

- **Source:** [US Accidents (2016–2023) on Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- **Size:** ~3 GB, 7+ million records across 49 US states
- **Subset used:** ~20% sample (~1.4M records) for model training and evaluation

## Project Structure

```
├── README.md
├── USAccidents_BigDataProject.ipynb   # Main notebook (end-to-end pipeline)
├── USAccidents_Report.pdf             # Project report
├── USAccidents_PPT.ppt                # Project presentation
├── Project_Proposal.pdf               # Original project proposal
```

## Setup & Prerequisites

### Requirements

- Python 3.8+
- Java 8 or 11 (required by Spark)
- Apache Spark 3.x
- A Kaggle account with an API token

### Python Dependencies

```bash
pip install pyspark kaggle matplotlib seaborn pandas
```

### Kaggle API Configuration

The first cell of the notebook downloads the dataset from Kaggle. Before running, open the notebook and replace the placeholder credentials with your own:

```python
kaggle_username = "YOUR_KAGGLE_USERNAME"
kaggle_key = "YOUR_KAGGLE_API_TOKEN"
```

> **Note:** Never commit your actual Kaggle credentials to a public repository. Consider using environment variables or a local `~/.kaggle/kaggle.json` file instead.

## Pipeline Overview

### 1. Data Ingestion & Cleaning

- Downloads and extracts the US Accidents CSV from Kaggle
- Selects relevant features: Severity, Start_Time, State, Temperature, Visibility, Wind_Speed, Weather_Condition, Sunrise_Sunset, Junction, Traffic_Signal, Pressure
- Handles missing values (median imputation for Pressure, zero-fill for Wind_Speed, drops critical nulls)
- Removes duplicates and invalid rows
- Saves cleaned data in Parquet format for efficient downstream processing

### 2. Feature Engineering

- **Temporal features:** Hour, DayOfWeek, Month, Year, is_rush_hour, is_weekend
- **Weather indicators:** is_rain, is_fog, is_snow (derived from Weather_Condition text)
- **Risk features:** low_visibility (Visibility < 5 mi), temp_visibility_interaction
- **Encoding:** StringIndexer for Sunrise_Sunset (sun_idx), boolean-to-integer casting for Junction and Traffic_Signal
- **Binary label:** Severity ≥ 3 mapped to label = 1 (high severity), otherwise 0

### 3. Exploratory Data Analysis

- Severity distribution and class imbalance analysis
- Hourly accident trends (morning and evening rush-hour peaks)
- Day vs. night accident comparison
- Year-wise severity rate trend (2016–2023) showing declining severity over time
- Monthly/seasonal severity patterns
- Top weather conditions by accident count
- State-level geographic analysis
- Infrastructure analysis (Junction, Traffic_Signal)
- Average visibility, temperature, and wind speed by severity level
- Outlier detection and data quality checks

### 4. State Severity Risk Index (SSRI)

A normalized risk metric ranking each state by the proportion of high-severity accidents (Severity ≥ 3). Higher SSRI values indicate a greater likelihood of severe crashes in that state.

### 5. Machine Learning Models

All models use an 80/20 train-test split with StandardScaler applied to Logistic Regression inputs. Class imbalance is addressed through multiple strategies.

**Random Forest:**
- Weighted RF (class weights inversely proportional to class frequency)
- RF with 1:1 downsampling
- 30 trees, maxDepth=8, maxBins=64

**Logistic Regression:**
- Baseline LR
- LR with 1:1 downsampling
- LR with 2:1 downsampling
- Class-weighted LR
- Threshold sweep (0.25–0.60) for optimal cutoff selection

### 6. Evaluation

Models are evaluated using ROC-AUC, PR-AUC, precision, recall, F1 score, and confusion matrices. A final comparison table and grouped bar chart visualize the precision-recall-F1 tradeoff across all six model variants.

## Key Findings

- **Severity is declining over time:** The proportion of high-severity accidents has dropped steadily from 2016 to 2023.
- **Rush hours are riskiest:** Accident frequency peaks during morning (7–9 AM) and evening (4–6 PM) commutes.
- **Fair weather dominates accident counts** due to higher traffic volume, though adverse conditions increase per-event risk.
- **Random Forest outperforms Logistic Regression** overall, with better ROC-AUC, PR-AUC, and a stronger balance between precision and recall.
- **LR with 1:1 downsampling** achieves the highest recall for severe accidents, making it useful for risk-sensitive applications where missing a severe case is costly.
- **Top predictive features** include traffic signals, visibility, pressure, temperature, and weather indicators (rain, fog, snow).

## How to Run

1. Clone the repository.
2. Install the dependencies listed above.
3. Open the notebook and replace the Kaggle credentials placeholder with your own username and API token.
4. Run all cells sequentially. The notebook handles data download, cleaning, EDA, SSRI computation, model training, evaluation, and visualization end to end.

## Technologies Used

- Apache Spark / PySpark / Spark SQL
- Spark MLlib (LogisticRegression, RandomForestClassifier, VectorAssembler, StandardScaler, StringIndexer)
- Matplotlib / Seaborn
- Parquet (intermediate storage)
- Kaggle API (data download)

## References

- Moosavi, Sobhan, et al. "A Countrywide Traffic Accident Dataset." 2019. [Kaggle Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
