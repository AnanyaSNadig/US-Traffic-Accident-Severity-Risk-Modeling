# US Traffic Accident Severity & Risk Modeling

A big-data-driven analysis of U.S. traffic accident records (2016–2023) using PySpark. The project builds a scalable analytics pipeline to identify patterns associated with high-severity accidents and predicts severity using machine learning models.

**Course:** CSGY-6513 Big Data | Section C | Spring 2026

**Instructor:** Prof. Amit Patel

**Team:**
- Ananya Nadig (an4968)
- Isha Jariwala (ij2221)
- Rhea Shastri (rs9459)

## Dataset

- **Source:** [US Accidents (2016–2023) on Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- **Size:** ~3 GB, 7+ million records across 49 US states
- **Subset used:** ~20% sample (~1.3M records) for model training and evaluation

## Project Structure

```
├── README.md
├── USAccidents_BigDataProject.ipynb                # Main notebook (end-to-end pipeline)
├── AIR_Final_Project_Report_6513C_SP2026.pdf       # Project report
├── Final ppt-Big data.pptx                         # Project presentation
└── Project_Proposal.pdf                            # Original project proposal
```

## Setup & Prerequisites

### Requirements

- Python 3.8+
- Java 8 or 11 (required by Spark)
- PySpark 3.x
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

## Pipeline Overview

### 1. Data Ingestion & Cleaning

- Downloads and extracts the US Accidents CSV from Kaggle using the Kaggle API
- Selects 11 relevant columns: Severity, Start_Time, State, Temperature, Visibility, Wind_Speed, Weather_Condition, Sunrise_Sunset, Junction, Traffic_Signal, Pressure
- Removes duplicates and drops rows with null values in critical columns
- Imputes Pressure with the median, fills missing Wind_Speed with 0
- Saves cleaned data in Parquet format for faster downstream processing

### 2. Feature Engineering

- **Temporal features:** Hour, DayOfWeek, Month, Year, is_rush_hour, is_weekend
- **Weather indicators:** is_rain, is_fog, is_snow (keyword-matched from Weather_Condition)
- **Risk features:** low_visibility (Visibility < 5 mi), temp_visibility_interaction
- **Encoding:** StringIndexer for Sunrise_Sunset, boolean-to-integer casting for Junction and Traffic_Signal
- **Binary label:** Severity ≥ 3 → label = 1 (high severity), otherwise 0

### 3. Exploratory Data Analysis

- Severity distribution and class imbalance analysis
- Hourly accident trends with morning and evening rush-hour peaks
- Day vs. night accident comparison
- Year-wise severity rate trend (2016–2023) showing a declining proportion of severe accidents
- Monthly severity patterns with peaks in late spring and early summer
- Top 10 weather conditions by accident count
- State-level geographic analysis and severity breakdown per state
- Road infrastructure analysis (Junction, Traffic_Signal)
- Average visibility, temperature, and wind speed by severity level
- Outlier detection and data quality checks

### 4. State Severity Risk Index (SSRI)

A custom metric that measures the proportion of high-severity accidents (Severity ≥ 3) within each state. SSRI normalizes for total accident volume, revealing hidden high-risk states that raw counts would miss. Rhode Island and Georgia lead with SSRI values around 0.47.

### 5. Machine Learning Models

All models use an 80/20 train-test split. StandardScaler is fitted on training data only and applied to Logistic Regression inputs. Class imbalance is addressed through multiple strategies.

**Random Forest (2 variants):**
- RF with class weighting (inversely proportional to class frequency)
- RF with 1:1 downsampling
- Configuration: 30 trees, maxDepth=8, maxBins=64

**Logistic Regression (4 variants):**
- Baseline LR
- LR with 1:1 downsampling
- LR with 2:1 downsampling
- Class-weighted LR
- Threshold sweep (0.25–0.60) for optimal cutoff selection

### 6. Evaluation

Models are evaluated using ROC-AUC, PR-AUC, severe-class precision, recall, F1 score, and confusion matrices. A final comparison table and grouped bar chart visualize the tradeoffs across all six model variants.

## Key Findings

- **Severity is declining over time:** The proportion of high-severity accidents dropped from ~35% in 2016 to under 5% in 2023.
- **Rush hours are riskiest:** Accident frequency peaks during morning (7–9 AM) and evening (4–6 PM) commutes.
- **Fair weather dominates accident counts** due to higher traffic volume, though adverse conditions increase per-event severity risk.
- **Random Forest outperforms Logistic Regression** with the highest F1 (0.397) and AUC (0.644), offering the best precision-recall balance.
- **LR with 1:1 downsampling** achieves the highest recall (0.922), making it suitable for risk-sensitive applications where missing a severe case is costly.
- **Top predictive features** include Traffic_Signal, Pressure, Wind_Speed, Temperature, and weather indicators.
- **SSRI reveals hidden hotspots:** States like Rhode Island and Georgia have the highest severity risk despite not leading in total accident counts.

## How to Run

1. Clone the repository.
2. Install the dependencies listed above.
3. Open the notebook and replace the Kaggle credential placeholders with your own username and API token.
4. Run all cells sequentially. The notebook handles data download, cleaning, EDA, SSRI computation, model training, evaluation, and visualization end to end.

## Technologies Used

- PySpark (Spark SQL, Spark MLlib)
- Parquet
- Kaggle API
- Matplotlib / Seaborn
- Pandas
- Python

## References

- Moosavi, Sobhan, et al. "A Countrywide Traffic Accident Dataset." 2019. [Kaggle Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
