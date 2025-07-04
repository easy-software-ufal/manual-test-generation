# Image2Test: Using ChatGPT to Build Manual Tests from Screenshots

This project provides tools and resources for generating, analyzing, and comparing manual test cases using GPT and other similarity metrics. It includes a Jupyter Notebook for generating test cases, a Streamlit application for analyzing test similarity, and CSV files containing human-generated test cases.

## Project Structure

### 1. Notebook
- **Path**: `notebook/img2test.ipynb`
- **Purpose**:
  - Generate manual test cases using GPT.
  - Save the generated test cases in Markdown format.
- **Key Features**:
  - Accepts a Google Drive link for an image.
  - Generates test cases based on a system and user prompt.
  - Saves the output along with metadata (system prompt, user prompt, and image link).

### 2. Streamlit Application
- **Path**: `main.py`
- **Purpose**:
  - Analyze the similarity of test cases using TF-IDF, BERT, and ROUGE-L metrics.
  - Provide both overall and row-by-row analysis of test cases.
- **Key Features**:
  - Upload HTML files containing test data.
  - Perform similarity analysis and visualize results.
  - Download analysis results as CSV files.

### 3. Human-Generated Test Cases
- **Path**: `human_tests_csv/`
- **Purpose**:
  - Store manually created test cases for various scenarios.
- **Files**:
  - `human - SEARCH.csv`: Test cases for search functionalities.
  - `human - MAPS.csv`: Test cases for map-related functionalities.
  - `human - EMAIL.csv`: Test cases for email functionalities.
  - `human - ECOMMERCE.csv`: Test cases for e-commerce functionalities.

## How to Use

### 1. Generate Test Cases
- Open the Jupyter Notebook: `notebook/generation_of_manual_tests_with_gpt.ipynb`.
- Provide a Google Drive link to an image and customize the prompts.
- Run the notebook to generate test cases and save them as Markdown files.

### 2. Analyze Test Cases
- Run the Streamlit application: `main.py`.
- Upload an HTML file containing test data.
- Use the "Overall Analysis" tab for summary metrics and visualizations.
- Use the "Row-by-Row Analysis" tab for detailed comparisons of individual test steps.

### 3. Compare with Human-Generated Tests
- Use the CSV files in `human_tests_csv/` as a reference for manually created test cases.
- Compare the generated test cases with these files to evaluate quality and coverage.

## Requirements

### Python Libraries
- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `beautifulsoup4`
- `sklearn`
- `sentence-transformers`
- `nltk`
- `rouge`

### Additional Tools
- Jupyter Notebook for running the test generation notebook.
- Streamlit for running the similarity analysis application.

## Future Enhancements
- Add support for more similarity metrics.
- Improve the visualization of analysis results.
- Automate the comparison of generated and human-created test cases.