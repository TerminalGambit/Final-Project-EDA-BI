# Exploratory Data Analysis Notebook - Setup Guide

## 📋 Overview

This directory contains the **Exploratory Data Analysis (EDA) notebook** that consolidates all analysis from scripts 01-03 into a single, interactive Jupyter notebook.

**Notebook:** `exploratory_data_analysis.ipynb`

---

## 📊 Data Requirements

### Expected Data Location

The notebook expects cleaned data at the following path (relative to the notebook):

```
../data/processed/cleaned_data.parquet
```

**Full project structure:**
```
final_pandas_x_sqlbi/
├── notebooks/
│   ├── exploratory_data_analysis.ipynb  ← The EDA notebook
│   └── README.md                         ← This file
├── data/
│   ├── raw/
│   │   └── all_data.csv                  ← Original raw data
│   └── processed/
│       └── cleaned_data.parquet          ← REQUIRED: Cleaned data
├── scripts/
│   └── 04_data_cleaning.py               ← Run this to generate cleaned data
└── ...
```

### If Cleaned Data Doesn't Exist

If `data/processed/cleaned_data.parquet` doesn't exist, you need to run the data cleaning script first:

```bash
# From project root directory
python scripts/04_data_cleaning.py
```

This will:
1. Load `data/raw/all_data.csv`
2. Clean and process the data
3. Save `data/processed/cleaned_data.parquet`

---

## 🔧 Installation Requirements

### Required Libraries

The notebook requires the following Python libraries:

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Static plotting
- **seaborn** - Statistical data visualization
- **plotly** - Interactive visualizations
- **pyarrow** or **fastparquet** - Parquet file support

### Installation Command

Install all required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn plotly pyarrow
```

**Or** if you prefer conda:

```bash
conda install pandas numpy matplotlib seaborn plotly pyarrow
```

### Using Project Requirements

If the project has a `requirements.txt` file at the root:

```bash
# From project root directory
pip install -r requirements.txt
```

---

## 🚀 Running the Notebook

### Step 1: Install Jupyter

If you don't have Jupyter installed:

```bash
pip install jupyter
```

Or use JupyterLab (recommended):

```bash
pip install jupyterlab
```

### Step 2: Launch Jupyter

**From the project root directory:**

```bash
# Launch Jupyter Notebook
jupyter notebook

# OR launch JupyterLab (recommended)
jupyter lab
```

This will open Jupyter in your browser.

### Step 3: Navigate and Run

1. Navigate to the `notebooks/` folder
2. Open `exploratory_data_analysis.ipynb`
3. Run cells sequentially using:
   - **Shift + Enter** - Run cell and move to next
   - **Ctrl + Enter** - Run cell and stay on current
   - **Cell → Run All** - Run all cells at once

---

## 📝 Notebook Contents

The notebook is organized into 5 main sections:

### 1. Setup & Data Loading
- Import libraries
- Load cleaned parquet data
- Display dataset overview

### 2. Initial Exploration
- Column analysis (dtypes, unique values, missing data)
- Missing data visualization
- Problematic columns identification
- Outlier detection with distribution plots

### 3. Data Structure Understanding
- Sample rows display
- Column-by-column interpretation
- Product-brand relationships
- Temporal coverage analysis

### 4. Detailed Analysis
- Brand statistics and duplicate investigation
- Price distribution by brand
- Stock distribution analysis
- Variant analysis (Default vs Custom)

### 5. Insights Summary
- Key findings
- Data quality observations
- Recommendations for further analysis

---

## ⚙️ Configuration

### Adjust Display Settings

If you want to modify pandas display options, edit the setup cell:

```python
# Configure pandas display
pd.set_option('display.max_columns', None)      # Show all columns
pd.set_option('display.max_rows', 100)          # Max rows to display
pd.set_option('display.width', None)            # Auto-detect width
pd.set_option('display.max_colwidth', 80)       # Column width limit
```

### Adjust Plot Sizes

To change default plot sizes, modify the setup cell:

```python
# Set plotting style
plt.rcParams['figure.figsize'] = (12, 6)  # Change to your preference
plt.rcParams['font.size'] = 10            # Change font size
```

---

## 🐛 Troubleshooting

### Issue: "No such file or directory: cleaned_data.parquet"

**Solution:** Run the data cleaning script first:
```bash
python scripts/04_data_cleaning.py
```

### Issue: "ModuleNotFoundError: No module named 'plotly'"

**Solution:** Install missing library:
```bash
pip install plotly
```

### Issue: "ModuleNotFoundError: No module named 'pyarrow'"

**Solution:** Install parquet support:
```bash
pip install pyarrow
# OR
pip install fastparquet
```

### Issue: Memory Error when loading data

**Solution:** The dataset is ~2.7 GB in memory. If you encounter memory issues:
1. Close other applications
2. Use a machine with more RAM
3. Or sample the data in the loading cell:
   ```python
   df = pd.read_parquet(DATA_PATH)
   df = df.sample(n=100000)  # Use sample for testing
   ```

### Issue: Plots not showing in Jupyter

**Solution:** Add this to the first cell:
```python
%matplotlib inline
```

---

## 📦 Dataset Information

**Expected Data Schema (10 columns):**

| Column           | Type       | Description                              |
|------------------|------------|------------------------------------------|
| product_id       | object     | URL slug identifier                      |
| title            | object     | Product name (French)                    |
| description      | object     | Product details (5.66% missing)          |
| brand            | object     | Brand name (24 unique)                   |
| offer_title      | object     | Variant name (e.g., color, size)         |
| offer_offer_id   | int64      | Shopify variant ID                       |
| offer_price      | float64    | Price in EUR                             |
| offer_in_stock   | bool       | Availability flag                        |
| stock            | int64      | Quantity available                       |
| date             | datetime64 | Timestamp of scrape                      |

**Dataset Size:**
- Rows: 1,724,927
- Memory: ~2.7 GB (in pandas)
- File size: ~1.5 MB (parquet compressed)

---

## 🎯 Next Steps After EDA

After completing the EDA, you can:

1. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run app/main.py
   ```

2. **Generate Metrics JSON:**
   ```bash
   python scripts/05_generate_metrics.py
   ```

3. **Explore with PyGWalker** (in the dashboard)

---

## 💡 Tips

- **Run cells in order** - Some cells depend on variables from previous cells
- **Use Kernel → Restart & Run All** to ensure reproducibility
- **Save your work frequently** with Ctrl+S or Cmd+S
- **Export as HTML** via File → Download as → HTML to share results
- **Add your own cells** to explore specific questions

---

## 📚 Additional Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Plotly Documentation](https://plotly.com/python/)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Jupyter Documentation](https://jupyter.org/documentation)

---

## ✅ Verification Checklist

Before running the notebook, verify:

- [ ] Python 3.7+ installed
- [ ] All required libraries installed (`pip install pandas numpy matplotlib seaborn plotly pyarrow`)
- [ ] Jupyter installed (`pip install jupyter`)
- [ ] Cleaned data exists at `../data/processed/cleaned_data.parquet`
- [ ] Running from the correct directory structure

---

**Questions or Issues?** Check the main project README.md or contact the project maintainer.
