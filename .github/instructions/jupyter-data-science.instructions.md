---
applyTo: notebooks/**/*.ipynb
---

# Jupyter Data Science Workflow
Instructions for working with Jupyter notebooks in the qda-bertopic project for qualitative data analysis using BERTopic and machine learning.

# Goal
Guide coding agents to effectively work with Jupyter notebooks for data science workflows, ensuring proper cell execution order, state management, memory optimization, and visualization rendering in this topic modeling project.

## Essentials
- **Sequential Execution**: Cells must be executed in order (top to bottom) as later cells depend on variables from earlier cells
- **State Awareness**: Use `globals().get('variable_name')` to safely check for variables from previous cells before using them
- **Memory Management**: Large datasets (>50k texts) require sampling and `calculate_probabilities=False` to prevent kernel crashes
- **Cell Editing**: Use `edit_notebook_file` tool with proper cellId, editType ('insert', 'edit', 'delete'), and newCode parameters
- **Cell Execution**: Always use `run_notebook_cell` tool (never run `jupyter notebook` commands in terminal)
- **Cell Summary**: Use `copilot_getNotebookSummary` to get cell IDs, types, languages, and execution status

## Tech Stack
- **[Jupyter Notebook](https://jupyter.org/)** - Interactive computing environment
- **[UV Package Manager](https://github.com/astral-sh/uv)** - Fast Python dependency management
- **[BERTopic](https://maartengr.github.io/BERTopic/)** - Topic modeling framework using transformers
- **[Pandas](https://pandas.pydata.org/)** (3.0.0+) - Data manipulation and analysis
- **[Sentence Transformers](https://www.sbert.net/)** (5.2.2+) - Text embeddings for semantic analysis
- **[UMAP](https://umap-learn.readthedocs.io/)** (0.5.11+) - Dimensionality reduction
- **[HDBSCAN](https://hdbscan.readthedocs.io/)** (0.8.41+) - Density-based clustering
- **[Plotly](https://plotly.com/python/)** - Interactive visualizations
- **[Matplotlib](https://matplotlib.org/)** & **[Seaborn](https://seaborn.pydata.org/)** - Static visualizations

## Project Structure
```
qda-bertopic/
├── notebooks/
│   └── qualitative_analysis.ipynb    # Main analysis workflow (11 cells)
├── data/
│   ├── raw/                          # Excel workbooks (gitignored)
│   ├── processed/                    # Extracted CSVs (gitignored)
│   └── results/                      # Analysis outputs (gitignored)
├── pyproject.toml                    # UV dependencies
└── uv.lock                           # Locked versions
```

## Key Files
- [notebooks/qualitative_analysis.ipynb](notebooks/qualitative_analysis.ipynb) - Main workflow with 11 cells (2 markdown, 9 code)
- [pyproject.toml](pyproject.toml) - Dependency definitions managed by UV

## Development Guidelines

### Working with Notebooks

#### Cell Execution Order
The main notebook has a strict execution order:
1. **Cell 1** (Markdown): Overview and instructions
2. **Cell 2** (Python): Import libraries and check installations
3. **Cell 3** (Python): Configure Excel file path
4. **Cell 4** (Python): Extract text data from Excel sheets
5. **Cell 5** (Python): Configure and initialize BERTopic model
6. **Cell 6** (Markdown): Topic modeling section header
7. **Cell 7** (Python): Fit model and extract topics
8. **Cell 8** (Python): Visualize topic distributions
9. **Cell 9** (Python): Create detailed topic view
10. **Cell 10** (Python): Generate visualizations (bar chart, hierarchy, heatmap)
11. **Cell 11** (Markdown): Export instructions

**Never skip cells or execute out of order** - this will cause `NameError` exceptions.

#### State Management Patterns
```python
# Good: Check if variable exists before using
if 'analysis_texts' not in globals():
    print("Error: Run previous cells first")
    raise ValueError("Missing required data")

# Good: Safely access previous results
model_choice = globals().get('model_choice', 'bertopic')

# Bad: Assume variables exist
print(f"Processing {len(analysis_texts)} texts")  # May fail if cell 4 wasn't run
```

#### Editing Cells
```python
# Get notebook structure first
copilot_getNotebookSummary(filePath='notebooks/qualitative_analysis.ipynb')

# Edit existing cell (preserve exact whitespace)
edit_notebook_file(
    filePath='notebooks/qualitative_analysis.ipynb',
    editType='edit',
    cellId='#VSC-fa1ba7c2',  # From summary
    language='python',
    newCode='import pandas as pd\nimport numpy as np\n# Updated imports'
)

# Insert new cell after specific cell
edit_notebook_file(
    filePath='notebooks/qualitative_analysis.ipynb',
    editType='insert',
    cellId='#VSC-8c237ca2',  # Insert after this cell
    language='python',
    newCode='# New analysis code\nprint("New cell")'
)

# Delete a cell
edit_notebook_file(
    filePath='notebooks/qualitative_analysis.ipynb',
    editType='delete',
    cellId='#VSC-563443bc'
)
```

#### Running Cells
```python
# Run a single cell
run_notebook_cell(
    filePath='notebooks/qualitative_analysis.ipynb',
    cellId='#VSC-fa1ba7c2',
    reason='Testing updated imports'
)

# Run cell and continue on error
run_notebook_cell(
    filePath='notebooks/qualitative_analysis.ipynb',
    cellId='#VSC-c4a69d0d',
    continueOnError=True
)

# Read cell output if truncated
read_notebook_cell_output(
    filePath='notebooks/qualitative_analysis.ipynb',
    cellId='#VSC-4c9af8b7'
)
```

### Memory Optimization

#### Large Dataset Handling
```python
# Automatic sampling for datasets > 50k texts
if len(analysis_texts) > 50000:
    print(f"⚠️ Dataset has {len(analysis_texts):,} texts")
    print("Sampling 50,000 texts to prevent memory issues...")
    sample_indices = np.random.choice(len(analysis_texts), 50000, replace=False)
    analysis_texts = [analysis_texts[i] for i in sorted(sample_indices)]
    text_sources = [text_sources[i] for i in sorted(sample_indices)]

# BERTopic configuration for large datasets
topic_model = BERTopic(
    nr_topics="auto",
    min_topic_size=10,  # Increase for large datasets to reduce topic count
    low_memory=True,    # Critical for >10k texts
    calculate_probabilities=False,  # Saves significant memory
    verbose=True
)
```

#### Memory-Efficient Operations
```python
# Good: Process in chunks for large visualizations
if len(topic_info) > 200:
    print("⚠️ Too many topics for visualization (>200)")
    print("Skipping hierarchy plot to prevent timeout")
else:
    fig = topic_model.visualize_hierarchy()
    _display_fig(fig, "Topic Hierarchy")

# Good: Disable multiprocessing in notebooks
from umap import UMAP
umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', n_jobs=1)
```

### Visualization Best Practices

#### Plotly Rendering
```python
# Always use HTML embedding for Plotly in notebooks
from IPython.display import display, HTML
import plotly.io as pio

# Set renderer at start of notebook
pio.renderers.default = 'notebook_connected'

# Helper function for consistent display
def _display_fig(fig, title=None):
    """Display Plotly figure as embedded HTML"""
    if title:
        print(f"\n{title}")
    html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    display(HTML(html))

# Use it for all Plotly figures
fig = topic_model.visualize_barchart(top_n_topics=15)
_display_fig(fig, "Top 15 Topics by Frequency")
```

#### Static Plots (Matplotlib/Seaborn)
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Configure for notebook
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 6)

# Always use context manager
with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots()
    # Plotting code
    plt.tight_layout()
    plt.show()
```

### Data Processing Patterns

#### Excel File Handling
```python
# Relative paths from notebook location
excel_file = '../data/raw/interview_data.xlsx'

# Handle missing files gracefully
import os
if not os.path.exists(excel_file):
    print(f"❌ Error: File not found: {excel_file}")
    print("\nPlace your Excel file in data/raw/ directory")
    raise FileNotFoundError(excel_file)

# Read all sheets
xl = pd.ExcelFile(excel_file)
print(f"📊 Found {len(xl.sheet_names)} sheets: {xl.sheet_names}")
```

#### Text Extraction
```python
# Default text column names
text_columns = ['text', 'response', 'comment', 'feedback', 
                'description', 'answer', 'input', 'narrative']

# Safe column detection
for sheet_name in xl.sheet_names:
    df = xl.parse(sheet_name)
    available_cols = [col for col in text_columns if col in df.columns]
    
    if not available_cols:
        print(f"⚠️ No text columns in '{sheet_name}'")
        print(f"Available: {list(df.columns)}")
        continue
    
    # Extract texts with source tracking
    for col in available_cols:
        texts = df[col].dropna().astype(str).tolist()
        sources = [f"{sheet_name}:{col}"] * len(texts)
        all_texts.extend(texts)
        all_sources.extend(sources)
```

### Error Handling

#### Common Issues & Solutions
```python
# Issue: Kernel crashes on large datasets
# Solution: Automatic sampling + low_memory mode
if len(texts) > 50000:
    texts = random.sample(texts, 50000)
topic_model = BERTopic(low_memory=True, calculate_probabilities=False)

# Issue: NameError for undefined variables
# Solution: Check globals before use
if 'topic_model' not in globals():
    raise ValueError("Run Cell 7 first to fit the model")

# Issue: Plotly figures don't render
# Solution: Use HTML embedding
html = fig.to_html(full_html=False, include_plotlyjs='cdn')
display(HTML(html))

# Issue: Empty text extraction
# Solution: Verify column names
print(f"Available columns: {list(df.columns)}")
print(f"Looking for: {text_columns}")
```

### Output Management

#### Saving Results
```python
# All outputs go to data/results/
output_dir = '../data/results/'
os.makedirs(output_dir, exist_ok=True)

# Consistent naming with model type
model_choice = 'bertopic'  # or 'top2vec'
theme_file = f'{output_dir}extracted_themes_{model_choice}.csv'
detail_file = f'{output_dir}detailed_themes_{model_choice}.csv'
assignment_file = f'{output_dir}text_topic_assignments_{model_choice}.csv'

# Save intermediate data
processed_dir = '../data/processed/'
pd.DataFrame({'text': texts, 'source': sources}).to_csv(
    f'{processed_dir}extracted_texts.csv', index=False
)
```

#### Print Progress
```python
# Good: Informative progress messages
print(f"📥 Loading Excel file: {os.path.basename(excel_file)}")
print(f"📊 Extracted {len(analysis_texts):,} texts from {len(set(text_sources))} sources")
print(f"🤖 Fitting {model_choice.upper()} model...")
print(f"✓ Identified {len(set(topics))} unique topics")
print(f"💾 Exported results to '{theme_file}'")

# Show samples for validation
print(f"\nSample text: {analysis_texts[0][:100]}...")
print(f"First 5 topics: {topics[:5]}")
```

## Reference Resources
- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [Sentence Transformers Models](https://www.sbert.net/docs/pretrained_models.html)
- [UMAP Parameters Guide](https://umap-learn.readthedocs.io/en/latest/parameters.html)
- [Jupyter Notebook Best Practices](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html)
- [UV Package Manager Docs](https://github.com/astral-sh/uv)
- [Plotly Python Documentation](https://plotly.com/python/)

## Common Workflows

### Starting a New Analysis
```bash
# 1. Place Excel file in data/raw/
cp ~/Downloads/survey_data.xlsx data/raw/

# 2. Start Jupyter
uv run jupyter lab

# 3. Open notebooks/qualitative_analysis.ipynb

# 4. Execute cells sequentially (1 → 11)
# - Cell 3: Update excel_file path
# - Cell 4: Verify text_columns match your data
# - Cell 5: Adjust min_topic_size if needed
```

### Modifying Analysis Parameters
```python
# Cell 5: Model configuration
min_topic_size = 5      # Increase to 10-15 for large datasets
nr_topics = "auto"       # Or specific number: 20
model_choice = 'bertopic'  # Keep as default

# Cell 7: Fitting options
topics, probs = topic_model.fit_transform(analysis_texts)

# Cell 10: Visualization options
fig_bar = topic_model.visualize_barchart(top_n_topics=15)  # Adjust top_n
```

### Debugging Cell Errors
```python
# 1. Get notebook summary to see execution status
copilot_getNotebookSummary(filePath='notebooks/qualitative_analysis.ipynb')

# 2. Check which cells have been executed
# Look for "Cell not executed" vs "Execution count = X"

# 3. Re-run from first failed cell
# If Cell 7 failed, restart kernel and run cells 2-7 in order

# 4. Check for common issues:
# - File paths (use ../ for relative paths)
# - Column names (print df.columns)
# - Memory (reduce dataset size or increase min_topic_size)
```

## Performance Tips
- **Large Datasets**: Increase `min_topic_size` to 15-20 to reduce topic count and memory usage
- **Faster Processing**: Use smaller embedding models (e.g., 'all-MiniLM-L6-v2' instead of default)
- **Better Topics**: Clean text data before analysis (remove IDs, standardize formatting)
- **Visualization**: Skip hierarchy plots if >200 topics (causes timeout)
- **Parallel Processing**: Disable in notebooks (set `n_jobs=1` in UMAP/HDBSCAN)
