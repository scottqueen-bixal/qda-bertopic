# AI-Enhanced Qualitative Data Analysis (QDA)

This project provides a complete workflow for processing qualitative data from Excel workbooks using AI-powered topic modeling. Extract themes and topics automatically using BERTopic or Top2Vec, then validate and refine them in your preferred qualitative data analysis tools.

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.8 or higher
- Excel workbook with qualitative data

### Step 1: Clone and Setup
```bash
# Navigate to your projects directory
cd /Users/scott.queen/Projects

# Clone or download this project
# (Assuming you have it locally already)

# Navigate to the project
cd QDALocal

# Create a virtual environment (recommended)
python -m venv qda_env
source qda_env/bin/activate

# Install dependencies
uv install -r requirements.txt
```

### Step 2: Prepare Your Data
1. **Place your Excel file** in the `data/raw/` directory
2. **Ensure your data has text columns** with names like:
   - `text`
   - `response`
   - `comment`
   - `feedback`
   - `description`

   *Or customize the column names in the notebook*

### Step 3: Run the Analysis
```bash
# Start Jupyter Notebook
jupyter notebook

# Or use JupyterLab (recommended)
jupyter lab
```

### Step 4: Open and Run the Notebook
1. In Jupyter, navigate to `notebooks/qualitative_analysis.ipynb`
2. **Update the file path** in Step 1 (around line 40):
   ```python
   excel_file = '../data/raw/your_workbook.xlsx'  # Change this to your file name
   ```
3. **Run all cells** in order (Shift+Enter or click the ▶️ button)
4. **Choose your model** in Step 3:
   ```python
   model_choice = 'bertopic'  # or 'top2vec'
   ```

### Step 5: Review Results
- **Visualizations**: Interactive topic maps and charts
- **Theme summaries**: Auto-generated topic descriptions
- **Export files**: Ready for import into QDA tools

## 📋 Detailed Getting Started Guide

### System Requirements
- **Python**: 3.8+
- **RAM**: 4GB minimum, 8GB+ recommended for large datasets
- **Storage**: 1GB free space
- **OS**: macOS, Windows, or Linux

### Installation Options

#### Option 1: Basic Installation (Recommended - No Build Issues)
```bash
# Use the basic requirements file (excludes Top2Vec which has build dependencies)
uv install -r requirements_basic.txt

# This installs BERTopic and all other dependencies without build issues
```

#### Option 2: Full Installation (May Have Build Issues)
```bash
# Try full installation (includes Top2Vec)
uv install -r requirements.txt

# If this fails with gensim build errors, use Option 1 above
```

#### Option 3: Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv qda_env

# Activate it
source qda_env/bin/activate  # macOS/Linux
# OR
qda_env\Scripts\activate     # Windows

# Install basic packages
uv install -r requirements_basic.txt
```

#### Option 4: Conda Environment
```bash
# Create conda environment
conda create -n qda python=3.9
conda activate qda

# Install packages (conda handles C extensions better)
conda install pandas openpyxl scikit-learn matplotlib seaborn plotly jupyter
uv install bertopic sentence-transformers umap-learn hdbscan
# Optional: uv install top2vec (may still have build issues)
```

### Data Preparation

#### Excel File Format
Your Excel workbook should contain:
- **Multiple sheets**: Each sheet becomes a separate data source
- **Text columns**: Qualitative responses, comments, or feedback
- **Clean data**: Remove special characters if possible

#### Example Data Structure
```
Sheet1: Customer_Survey
| ID | Age | Gender | response                          |
|----|-----|--------|-----------------------------------|
| 1  | 25  | F      | The service was excellent...     |
| 2  | 32  | M      | I found the interface confusing...|

Sheet2: Employee_Feedback
| employee_id | department | comment                           |
|-------------|------------|-----------------------------------|
| 101         | Sales      | Management support is great...   |
```

#### Supported Column Names
The notebook automatically detects these column names:
- `text`
- `response`
- `comment`
- `feedback`
- `description`
- `answer`
- `input`

*Add custom column names to the `text_columns` list in the notebook*

### Running the Analysis

#### Step-by-Step Execution

1. **Cell 1-2**: Install and import libraries
   ```python
   # Run this if packages aren't installed
   !uv install -r ../requirements.txt
   ```

2. **Cell 3**: Load Excel workbook
   - Update the `excel_file` path
   - Check that sheets loaded correctly

3. **Cell 4**: Extract text data
   - Review which columns were found
   - Check text extraction summary

4. **Cell 5**: Choose and run AI model
   - Set `model_choice = 'bertopic'` or `'top2vec'`
   - BERTopic: Better visualizations, hierarchical topics
   - Top2Vec: Faster, good for large datasets

5. **Cell 6**: Generate visualizations
   - Topic maps, hierarchies, and charts
   - Interactive plots (may require browser)

6. **Cell 7**: Export results
   - CSV files saved to `data/results/`
   - Ready for QDA tool import

#### Model Selection Guide

| Model | Best For | Pros | Cons | Availability |
|-------|----------|------|------|--------------|
| BERTopic | Most QDA scenarios | Rich visualizations, hierarchical topics, probability scores | Slower on large datasets | ✅ Always available |
| Top2Vec | Large datasets, speed | Fast processing, good clustering | Limited visualizations | ⚠️ Optional (build issues) |

**Note**: If Top2Vec installation fails, the notebook will automatically fall back to BERTopic, which provides excellent results for most qualitative analysis tasks.

### Understanding the Output

#### Generated Files
```
data/results/
├── extracted_themes_bertopic.csv          # Topic summaries
├── detailed_themes_bertopic.csv           # Detailed topics with word scores
├── text_topic_assignments_bertopic.csv    # Individual text assignments
├── extracted_texts.csv                    # Processed text data
```

#### Visualization Types
- **Topic Map**: 2D visualization of topic relationships
- **Topic Hierarchy**: Tree structure showing topic relationships
- **Topic Distribution**: Bar chart of topic frequencies
- **Barchart**: Top words per topic

### Importing into QDA Tools

#### Taguette
1. Open Taguette
2. Import CSV files as documents
3. Use extracted themes as initial tags
4. Manually validate and refine

#### QualCoder
1. Create new project
2. Import texts from CSV
3. Use topic assignments for initial coding
4. Review and adjust codes

#### NVivo/ATLAS.ti
1. Create new project
2. Import CSV files
3. Create nodes based on discovered topics
4. Use detailed theme information for code descriptions

### Troubleshooting

#### Common Issues

**"File not found" error**
```python
# Check your file path
excel_file = '../data/raw/your_workbook.xlsx'
# Make sure the file exists in data/raw/
```

**"No texts extracted"**
```python
text_columns = ['text', 'response', 'comment', 'your_column_name']
```

**Memory errors**

**Poor topic quality**

**Visualization not showing**

#### Performance Optimization

**For Large Datasets (>10,000 texts):**
- Use Top2Vec instead of BERTopic
- Increase `min_topic_size` to 10-15
- Consider Databricks Community Edition

**For Better Topics:**
- Preprocess text (remove stop words, lemmatize)
- Experiment with different `min_topic_size` values
- Try both models and compare results

### Advanced Configuration

#### Customizing BERTopic
```python
topic_model = BERTopic(
    language="english",           # Change for other languages
    calculate_probabilities=True, # Get confidence scores
    verbose=True,                 # Show progress
    min_topic_size=5,            # Minimum texts per topic
    nr_topics="auto"             # Or set specific number
)
```

#### Customizing Top2Vec
```python
topic_model = Top2Vec(
    documents=all_texts,
    speed="learn",               # 'fast-learn', 'learn', 'deep-learn'
    workers=4                    # CPU cores to use
)
```

#### Adding Text Preprocessing
Add this before running the model:
```python
import re
from nltk.corpus import stopwords

def preprocess_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stop words (optional)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

all_texts = [preprocess_text(text) for text in all_texts]
```

### Self-Hosting Options

#### Databricks Community Edition
1. Sign up at [community.databricks.com](https://community.databricks.com)
2. Upload your notebook
3. Run with scalable clusters for large datasets

#### Google Colab
1. Upload notebook to Google Drive
2. Open with Colab
3. Install dependencies: `!uv install -r requirements.txt`
4. Mount Drive for data access

#### Local Server
```bash
# Run Jupyter on a specific port
jupyter notebook --port=8889

# Or with no browser
jupyter notebook --no-browser --port=8889
```

### Project Structure
```
QDALocal/
├── README.md                    # This guide
├── requirements.txt             # Python dependencies
├── notebooks/
│   └── qualitative_analysis.ipynb  # Main analysis notebook
├── data/
│   ├── raw/                     # Your Excel files go here
│   ├── processed/               # Extracted texts (auto-generated)
│   └── results/                 # Analysis outputs (auto-generated)
└── .gitignore                   # Git ignore rules
```

### Dependencies
- `pandas`: Data manipulation
- `openpyxl`: Excel file reading
- `bertopic`: Topic modeling (recommended)
- `top2vec`: Alternative topic modeling *(optional - may have build issues)*
- `jupyter`: Notebook interface
- `matplotlib`/`seaborn`/`plotly`: Visualizations
- `scikit-learn`: Machine learning utilities
- `sentence-transformers`: Text embeddings
- `umap-learn`/`hdbscan`: Dimensionality reduction

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### License
This project is open source. Please cite appropriately if used in research.

### Support
For issues or questions:
1. Check the troubleshooting section above
2. Review the notebook comments
3. Test with sample data first

---

**Last updated**: January 28, 2026</content>
<parameter name="filePath">/Users/scott.queen/Projects/QDALocal/README.md
