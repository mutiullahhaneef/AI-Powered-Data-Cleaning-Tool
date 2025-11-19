# AI Data Cleaning & Analysis App

A Streamlit-based web application designed to simplify data cleaning and preprocessing.
The app enables users to upload datasets, detect quality issues, apply cleaning operations, visualize data, and export the final cleaned dataset.

This tool is suitable for students, data analysts, researchers, and anyone who works with messy datasets and needs a quick, efficient cleaning workflow.

---

## Overview

This application provides an end-to-end data cleaning pipeline, allowing users to:

* Upload CSV datasets
* Explore data structure and statistics
* Detect missing values, duplicates, and outliers
* Apply data cleaning operations
* Generate visual summaries
* Compare data before and after cleaning
* Download the cleaned dataset

---

## Features

### 1. Upload & Preview

* Import any CSV file
* View dataset shape and metadata
* Display column types and descriptive statistics
* Preview top rows for quick inspection

### 2. Data Issue Detection

* Missing value summary
* Duplicate row detection
* Outlier detection using Interquartile Range (IQR)
* Visual reports including bar charts and boxplots

### 3. Data Cleaning Tools

* Drop missing values
* Fill missing values (mean, median, mode)
* Remove duplicate rows
* Handle outliers

### 4. AI-Style Suggestions

The system analyzes the dataset and generates recommendations on how to clean it based on detected issues.

### 5. Export Cleaned Data

* One-click download of the processed dataset

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ai-data-cleaner.git
cd ai-data-cleaner
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Project Structure

```
📦 ai-data-cleaner
├── app.py
├── README.md
├── requirements.txt
└── sample_data/        (optional)
```

---

## Requirements

Ensure the following packages are included in `requirements.txt`:

* streamlit
* pandas
* numpy
* matplotlib
* seaborn

---

## Deployment

This app can be deployed on any Streamlit-supported hosting platform:

* Streamlit Cloud
* Render
* HuggingFace Spaces
* Local or LAN deployment

Use the same command for deployment:

```bash
streamlit run app.py
```

---

## Contributing

Contributions are welcome.
If you would like to enhance performance, improve UI, expand features, or fix issues, feel free to open a pull request or create an issue.
