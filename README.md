# 🧹 AI-Powered Data Cleaning Tool

An intelligent, interactive web application built with Streamlit that automates the data cleaning process using AI-powered suggestions and comprehensive data quality analysis.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **🔍 Automatic Data Quality Detection**
  - Missing value identification and analysis
  - Duplicate row detection
  - Outlier detection using IQR method
  - Data type analysis and optimization suggestions

- **🤖 AI-Powered Recommendations**
  - Intelligent cleaning suggestions based on data patterns
  - Severity-based issue prioritization
  - Context-aware imputation strategies
  - Cardinality analysis for categorical optimization

- **📊 Interactive Visualizations**
  - Missing value distribution charts
  - Outlier detection boxplots
  - Before/after comparison metrics
  - Real-time data previews

- **⚡ One-Click Cleaning**
  - Automated cleaning pipeline
  - Customizable cleaning strategies
  - Detailed cleaning operation logs
  - Memory usage optimization

- **📥 Export Capabilities**
  - Download cleaned datasets as CSV
  - Preserve data integrity
  - Maintain original data backup

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-data-cleaner.git
cd ai-data-cleaner
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## 📦 Dependencies

- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## 🎯 Usage

### Upload Your Data
1. Click on the sidebar file uploader
2. Select a CSV file from your computer
3. Or use one of the pre-loaded sample datasets

### Explore Data Quality
Navigate through the tabs to:
- **Data Overview**: View basic statistics and data types
- **Data Issues**: Identify missing values, duplicates, and outliers
- **AI Suggestions**: Get intelligent recommendations for cleaning
- **Cleaning Actions**: Apply automated or manual cleaning operations
- **Before/After**: Compare original vs cleaned data

### Customize Cleaning Options
Use the sidebar to configure:
- Missing value handling strategy (drop, fill with mean/median/mode)
- Duplicate removal preferences
- Outlier treatment methods (cap or remove)

### Apply Cleaning
1. Review AI suggestions
2. Configure cleaning preferences in sidebar
3. Click "Apply Automated Cleaning" button
4. Review cleaning log and results
5. Download cleaned dataset

## 📊 Sample Datasets

The tool includes three sample datasets for testing:

1. **Employee Data**: HR dataset with salary, performance, and demographic information
2. **Sales Data**: E-commerce transactions with product, customer, and rating data
3. **Healthcare Data**: Patient records with vital signs and health metrics

## 🛠️ Core Functionality

### Missing Value Detection
- Identifies columns with null values
- Calculates missing percentages
- Visualizes distribution of missing data

### Duplicate Detection
- Finds exact duplicate rows
- Provides count and percentage metrics

### Outlier Detection
- Uses Interquartile Range (IQR) method
- Calculates upper and lower bounds
- Identifies outlier counts per column

### Data Cleaning Strategies
- **Drop**: Remove rows with missing values
- **Fill**: Impute using mean, median, or mode
- **Cap**: Limit outliers to IQR bounds
- **Remove**: Delete outlier rows

## 🎨 Screenshots

*Upload your screenshots here showing the different tabs and features*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Data manipulation powered by [Pandas](https://pandas.pydata.org/)
- Visualizations created with [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)

## 📧 Contact

Your Name - [Mutiullah Haneef](https://www.linkedin.com/in/mutiullah-haneef-3311922b5/)

Project Link: [AI-Powered-Data-Cleaning-Tool](https://github.com/mutiullahhaneef/AI-Powered-Data-Cleaning-Tool)

## 🗺️ Roadmap

- [ ] Add support for Excel files
- [ ] Implement advanced ML-based anomaly detection
- [ ] Add data profiling reports
- [ ] Support for larger datasets with chunking
- [ ] Export to multiple formats (JSON, Parquet)
- [ ] Add data transformation features
- [ ] Integrate with cloud storage services

## ⚠️ Known Issues

- Large datasets (>100MB) may cause performance issues
- Complex nested data structures are not fully supported
- Some special characters in column names may cause issues

## 💡 Tips

- Start with sample data to understand the tool's capabilities
- Review AI suggestions before applying automated cleaning
- Always download and compare the cleaned data with your original
- Use the cleaning log to understand what operations were performed
