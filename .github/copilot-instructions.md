# Copilot Instructions for QDA-BERTopic

## Project Overview
This is an AI-enhanced qualitative data analysis (QDA) tool that processes Excel workbooks containing qualitative text data and uses BERTopic (or optionally Top2Vec) to automatically extract themes and topics. The output is designed for validation and import into traditional QDA tools like Taguette, QualCoder, NVivo, and ATLAS.ti.

## Tech Stack
- **Language**: Python 3.12
- **Package Manager**: uv (UV package manager for fast dependency resolution)
- **Main Framework**: Jupyter Notebook for interactive analysis
- **Core Libraries**:
  - `bertopic` (0.17.4+) - Primary topic modeling framework
  - `pandas` (3.0.0+) - Data manipulation
  - `openpyxl` (3.1.5+) - Excel file reading
  - `sentence-transformers` (5.2.2+) - Text embeddings
  - `umap-learn` (0.5.11+), `hdbscan` (0.8.41+) - Dimensionality reduction/clustering
  - `matplotlib`, `seaborn`, `plotly` - Visualizations
  - `scikit-learn` (1.8.0+) - ML utilities

## Project Structure
```
qda-bertopic/
├── .github/
│   ├── prompts/          # Agent prompt templates
│   └── pull_request_template.md
├── data/
│   ├── raw/             # Excel workbooks go here (gitignored)
│   ├── processed/       # Extracted text CSVs (gitignored)
│   └── results/         # Analysis outputs (gitignored)
├── notebooks/
│   └── qualitative_analysis.ipynb  # Main analysis workflow
├── pyproject.toml       # UV/uv dependencies
├── uv.lock             # Locked dependencies
└── README.md           # Comprehensive user guide
```

## Development Workflow

### Environment Setup
```bash
# Always use uv for dependency management
uv sync                 # Install all dependencies from pyproject.toml
uv add <package>        # Add new dependency
uv run jupyter lab      # Run jupyter without activating venv
```

**DO NOT**:
- Use `pip install` directly (use `uv add` instead)
- Create manual requirements.txt files (uv manages dependencies via pyproject.toml)
- Install packages globally

### Running the Notebook
```bash
# Start Jupyter Lab
uv run jupyter lab

# Or Jupyter Notebook
uv run jupyter notebook
```

### Key Configuration Points
1. **Excel file path** (Cell 3): Update `excel_file = '../data/raw/your_file.xlsx'`
2. **Text column names** (Cell 4): Modify `text_columns = ['text', 'response', 'comment', ...]`
3. **Model choice** (Cell 5): Set `model_choice = 'bertopic'` (Top2Vec is optional and may have build issues)
4. **Memory optimization** (Cell 5): Large datasets (>50k texts) are automatically sampled

## Coding Guidelines

### Jupyter Notebook Best Practices
- **Cell order matters**: Cells must be executed sequentially (1→7)
- **State management**: Use `globals().get()` to safely check for variables from previous cells
- **Large datasets**: Automatic sampling at 50k texts to prevent memory issues
- **Error handling**: Always wrap file operations in try-except blocks
- **Memory efficiency**: `calculate_probabilities=False` for large datasets

### Python Style
- Use descriptive variable names (e.g., `analysis_texts`, `text_sources`)
- Add progress prints for long operations: `print(f"Processing {len(data)} items...")`
- Display sample outputs: `print(f"Sample text: {text[:100]}...")`
- Always save intermediate results to CSV in `data/processed/`

### Data Processing Patterns
```python
# Good: Safely handle missing columns
available_cols = [col for col in expected_cols if col in df.columns]

# Good: Track data sources
text_sources = [f"{sheet_name}:{col}" for ...]

# Good: Handle large datasets
if len(texts) > 50000:
    sample_indices = np.random.choice(len(texts), 50000, replace=False)
```

### Visualization Patterns
```python
# Always use HTML embedding for Plotly in notebooks
from IPython.display import display, HTML

def _display_fig(fig, title=None):
    html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    display(HTML(html))

# Skip visualizations if too many topics
if len(theme_info) >= 200:
    print('Skipping hierarchy: too many topics')
```

## Common Pitfalls & Workarounds

### Memory Issues
- **Problem**: Large datasets cause kernel crashes
- **Solution**: Automatic sampling at 50k texts (implemented in Cell 5)
- **Solution**: `low_memory=True` flag in BERTopic
- **Solution**: `calculate_probabilities=False` to reduce memory

### Top2Vec Build Failures
- **Problem**: gensim dependency has C extension build issues
- **Solution**: Top2Vec is optional; BERTopic is the primary model
- **Workaround**: If Top2Vec needed, try conda: `conda install gensim`

### Excel File Issues
- **Problem**: File not found errors
- **Solution**: Always use relative paths from notebook: `../data/raw/file.xlsx`
- **Check**: Ensure file is in `data/raw/` directory

### No Texts Extracted
- **Problem**: Column names don't match
- **Solution**: Print available columns and update `text_columns` list
- **Default columns**: text, response, comment, feedback, description, answer, input

### Visualization Rendering Issues
- **Problem**: Plotly figures don't display or block kernel
- **Solution**: Use HTML embedding via `_display_fig()` function (already implemented)
- **Solution**: Set `pio.renderers.default = 'notebook_connected'`

## File Operations

### Input Files
- Place Excel workbooks (.xlsx, .xls) in `data/raw/`
- Multi-sheet workbooks are fully supported
- Text data can be in any column (configure in notebook)

### Output Files
Generated in `data/results/`:
- `extracted_themes_bertopic.csv` - Topic summaries
- `detailed_themes_bertopic.csv` - Detailed topics with word scores
- `text_topic_assignments_bertopic.csv` - Text-to-topic mappings
- `extracted_texts.csv` (in `data/processed/`) - All extracted texts

### Git Ignore Rules
- All data files (raw/*.xlsx, processed/*.csv, results/*.csv) are gitignored
- Keep data/ structure via .gitkeep files
- Never commit actual data files

## Testing & Validation

### Manual Testing Checklist
1. Test with small dataset (< 1000 texts) first
2. Verify text extraction: Check `extracted_texts.csv`
3. Validate topics: Review `theme_info` output
4. Check visualizations render without blocking
5. Confirm exports exist in `data/results/`

### Common Test Scenarios
- **Empty Excel sheets**: Should print warning, continue processing other sheets
- **Missing columns**: Should list available columns, skip sheet
- **Large datasets**: Should auto-sample to 50k with message
- **Single topic**: Should handle gracefully (no hierarchy viz)

## Performance Optimization

### For Large Datasets (>10k texts)
- Increase `min_topic_size` to 10-15 (reduces topic count and memory)
- Use `nr_topics="auto"` (already set)
- Disable multiprocessing: `n_jobs=1` in UMAP/HDBSCAN
- Consider running on Databricks Community Edition

### For Better Topic Quality
- Preprocess text (remove noise, standardize)
- Adjust `min_topic_size` (5 for more topics, 15 for fewer)
- Ensure text quality: meaningful content, not IDs or codes

## Key Dependencies Notes

### BERTopic Configuration
- Always use `low_memory=True` for datasets > 10k texts
- Disable probabilities for memory efficiency
- Use single-threaded processing (`n_jobs=1`) to avoid multiprocessing issues

### UMAP/HDBSCAN
- Disable parallel processing in notebook environments
- Core settings already optimized in notebook

## Making Changes

### Adding New Dependencies
```bash
# Add package via uv
uv add package-name

# Update notebook cell 2 comment to reflect new requirement
```

### Modifying Analysis Logic
- Keep cell structure intact (7 main cells)
- Add helper functions before first usage
- Document parameter changes in markdown cells

### Adding New Export Formats
- Create new export function in Cell 7
- Follow pattern: `output_file = f"../data/results/name_{model_choice}.ext"`
- Print confirmation: `✓ Exported to 'path'`

## Useful Commands

```bash
# Check Python version (must be 3.12+)
python --version

# Verify uv installation
uv --version

# List installed packages
uv pip list

# Run notebook
uv run jupyter lab

# Check data directory structure
ls -la data/raw/ data/processed/ data/results/
```

## Additional Resources
- README.md contains comprehensive user guide
- Notebook has detailed inline documentation
- All cells include error handling and user-friendly messages
