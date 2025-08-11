# flake8: noqa
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from rouge import Rouge

# Download nltk data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)

download_nltk_data()

st.set_page_config(
    page_title="Test Similarity Analysis",
    page_icon="📊",
    layout="wide"
)

# Functions for data extraction and analysis
def extract_test_data_from_html(file_path):
    """Extract test data including test name, type, and table contents from the HTML file."""
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    test_data = {}
    containers = soup.find_all("div", class_="container")  # Find all containers

    for container in containers:
        # Get the test name from the <h1> tag
        test_name = container.find("h1").text.strip()

        # Get all boxes for this test
        boxes = container.find_all("div", class_="box")
        for box in boxes:
            # Get the test type from the <p> tag inside .type-test
            test_type = box.find("div", class_="type-test").find("p").text.strip()

            # Extract table data
            rows = box.find_all("tr")
            table_data = []
            for row in rows[1:]:  # Skip the header
                cols = row.find_all("td")
                if cols:
                    table_data.append([col.text.strip() for col in cols])

            # Convert table data to a DataFrame
            if table_data:
                df = pd.DataFrame(table_data, columns=["#", "Actions", "Verifications"])

            # Organize data by test name and type
            if test_name not in test_data:
                test_data[test_name] = {}
            test_data[test_name][test_type] = df

    return test_data

def handle_null_values(text1, text2):
    """Handle null/None values in text comparison."""
    if pd.isna(text1) and pd.isna(text2):
        return 1.0  # Both empty means they're identical
    elif pd.isna(text1) or pd.isna(text2):
        return 0.0  # One empty means no similarity
    return None  # Both non-null, proceed with normal comparison

# Cosine similarity using TF-IDF Vectorizer
def cosine_similarity_tfidf(text1, text2):
    """Calculate TF-IDF cosine similarity with null value handling."""
    # First check for null values
    null_result = handle_null_values(text1, text2)
    if null_result is not None:
        return null_result

    # Convert to strings to handle non-string inputs
    text1 = str(text1)
    text2 = str(text2)

    # Handle empty strings
    if not text1.strip() or not text2.strip():
        return 0.0

    try:
        vectorizer = TfidfVectorizer().fit([text1, text2])
        vectors = vectorizer.transform([text1, text2])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    except Exception:
        # If there's an error in calculation, return 0
        return 0.0

# Semantic similarity using Sentence-BERT
def semantic_similarity_bert(text1, text2):
    """Calculate semantic similarity with null value handling."""
    # First check for null values
    null_result = handle_null_values(text1, text2)
    if null_result is not None:
        return null_result

    # Convert to strings to handle non-string inputs
    text1 = str(text1)
    text2 = str(text2)

    # Handle empty strings
    if not text1.strip() or not text2.strip():
        return 0.0

    try:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embeddings = model.encode([text1, text2])
        return np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    except Exception:
        # If there's an error in calculation, return 0
        return 0.0

def calculate_rouge_l(text1, text2):
    """Calculate ROUGE-L score between two texts."""
    # Handle null values
    null_result = handle_null_values(text1, text2)
    if null_result is not None:
        return null_result

    # Convert to strings to handle non-string inputs
    text1 = str(text1)
    text2 = str(text2)

    # Rouge expects non-empty strings
    if not text1.strip() or not text2.strip():
        return 0.0

    try:
        # Initialize Rouge
        rouge = Rouge()
        # Calculate ROUGE scores
        scores = rouge.get_scores(text1, text2)
        return scores[0]['rouge-l']['f']
    except Exception:
        # If there's an error in ROUGE calculation, return 0
        return 0.0

def count_words(text):
    """Count the number of words in a text."""
    if pd.isna(text) or text == "":
        return 0
    return len(word_tokenize(str(text)))

def perform_overall_analysis(test_data, progress_bar=None):
    """Perform overall analysis for all tests with TF-IDF, BERT, and ROUGE-L."""
    analysis_results = []
    counter = 0
    for test_name, types in test_data.items():
        if len(types) == 2:
            type_a = list(types.keys())[0]
            type_b = list(types.keys())[1]

            df_a = types[type_a]
            df_b = types[type_b]

            # Calculated metrics for each method
            tfidf_scores = []
            bert_scores = []
            rouge_scores = []

            # Use the shorter DataFrame length to avoid index errors
            total_rows = min(len(df_a), len(df_b))

            if progress_bar and total_rows > 0:
                progress_bar.progress(counter / len(test_data))
                counter += 1


            for i in range(total_rows):
                row_a = df_a.iloc[i]
                row_b = df_b.iloc[i]

                action_a = row_a["Actions"]
                verification_a = row_a["Verifications"]
                action_b = row_b["Actions"]
                verification_b = row_b["Verifications"]

                # Combined text for overall scoring
                text_a = str(action_a) + " " + str(verification_a)
                text_b = str(action_b) + " " + str(verification_b)

                # Calculate scores with each method
                # TF-IDF scores (one for actions, one for verifications)
                tfidf_action = cosine_similarity_tfidf(action_a, action_b)
                tfidf_verification = cosine_similarity_tfidf(verification_a, verification_b)
                tfidf_combined = (tfidf_action + tfidf_verification) / 2
                tfidf_scores.append(tfidf_combined)

                # BERT scores (one for actions, one for verifications)
                bert_action = semantic_similarity_bert(action_a, action_b)
                bert_verification = semantic_similarity_bert(verification_a, verification_b)
                bert_combined = (bert_action + bert_verification) / 2
                bert_scores.append(bert_combined)

                # ROUGE-L score (for combined text)
                rouge_score = calculate_rouge_l(text_a, text_b)
                rouge_scores.append(rouge_score)

            # Calculate averages
            avg_tfidf = np.mean(tfidf_scores) if tfidf_scores else 0
            avg_bert = np.mean(bert_scores) if bert_scores else 0
            avg_rouge = np.mean(rouge_scores) if rouge_scores else 0

            # Row counts
            rows_a = len(df_a)
            rows_b = len(df_b)

            # Add to results
            analysis_results.append({
                "Test Name": test_name,
                # "Type A": type_a,
                # "Type B": type_b,
                "Rows A": rows_a,
                "Rows B": rows_b,
                "TF-IDF Similarity": round(avg_tfidf, 4),
                "BERT Similarity": round(avg_bert, 4),
                "ROUGE-L Score": round(avg_rouge, 4)
            })

    return pd.DataFrame(analysis_results)

def perform_row_by_row_analysis(test_name, test_data, progress_bar=None):
    """Perform row-by-row analysis for a specific test with all similarity metrics."""
    if test_name not in test_data or len(test_data[test_name]) != 2:
        return None

    types = test_data[test_name]
    type_a = list(types.keys())[0]
    type_b = list(types.keys())[1]

    df_a = types[type_a]
    df_b = types[type_b]

    # Determine number of rows to compare
    rows_to_compare = min(len(df_a), len(df_b))

    row_analysis_results = []

    # Set up progress bar if provided
    if progress_bar and rows_to_compare > 0:
        progress_steps = 100 / rows_to_compare
        progress_values = [i * progress_steps/100 for i in range(rows_to_compare)]
    else:
        progress_values = [0] * rows_to_compare

    for i in range(rows_to_compare):
        row_a = df_a.iloc[i]
        row_b = df_b.iloc[i]

        # Update progress bar if provided
        if progress_bar:
            progress_bar.progress(progress_values[i])

        # Extract text from both rows
        action_a = str(row_a["Actions"])
        verification_a = str(row_a["Verifications"])
        action_b = str(row_b["Actions"])
        verification_b = str(row_b["Verifications"])

        # Combined text for overall analysis
        text_a = action_a + " " + verification_a
        text_b = action_b + " " + verification_b

        # Calculate word counts
        action_a_words = count_words(action_a)
        verification_a_words = count_words(verification_a)
        action_b_words = count_words(action_b)
        verification_b_words = count_words(verification_b)

        total_words_a = action_a_words + verification_a_words
        total_words_b = action_b_words + verification_b_words
        word_diff = abs(total_words_a - total_words_b)

        # Calculate similarities with all methods
        # TF-IDF
        tfidf_action = cosine_similarity_tfidf(action_a, action_b)
        tfidf_verification = cosine_similarity_tfidf(verification_a, verification_b)
        tfidf_combined = (tfidf_action + tfidf_verification) / 2

        # BERT
        bert_action = semantic_similarity_bert(action_a, action_b)
        bert_verification = semantic_similarity_bert(verification_a, verification_b)
        bert_combined = (bert_action + bert_verification) / 2

        # ROUGE-L
        rouge_score = calculate_rouge_l(text_a, text_b)

        # Add to results
        row_analysis_results.append({
            "Row #": i + 1,
            "Words A": total_words_a,
            "Words B": total_words_b,
            "Word Diff": word_diff,
            "TF-IDF Actions": round(tfidf_action, 4),
            "TF-IDF Verif": round(tfidf_verification, 4),
            "TF-IDF Combined": round(tfidf_combined, 4),
            "BERT Actions": round(bert_action, 4),
            "BERT Verif": round(bert_verification, 4),
            "BERT Combined": round(bert_combined, 4),
            "ROUGE-L": round(rouge_score, 4)
        })

    result_df = pd.DataFrame(row_analysis_results)

    return {
        "type_a": type_a,
        "type_b": type_b,
        "dataframe": result_df
    }

# Streamlit application


st.title("Test Similarity Analysis Tool")
st.write("Compare test similarity using TF-IDF, BERT, and ROUGE-L metrics")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Overall Analysis", "Row-by-Row Analysis", "About"])

with tab1:
    st.header("Overall Test Analysis")
    st.write("Upload your HTML file to see overall test similarity metrics across different methods")

    uploaded_file = st.file_uploader("Upload the 'tables.html' file", type=["html"], key="overall_uploader")

    if uploaded_file:
        # Save uploaded file temporarily
        temp_path = Path("uploaded_tables.html")
        temp_path.write_bytes(uploaded_file.read())

        # Extract test data
        test_data = extract_test_data_from_html(temp_path)

        if st.button("Generate Overall Analysis"):
            # Create progress indicators
            spinner_placeholder = st.empty()
            progress_placeholder = st.empty()

            with spinner_placeholder.container():
                with st.spinner("Calculating similarity metrics..."):
                    progress_bar = progress_placeholder.progress(0)

                    # Perform overall analysis
                    overall_df = perform_overall_analysis(test_data, progress_bar)

                    # Clear progress indicators
                    progress_placeholder.empty()

            st.success("✅ Analysis complete!")

            # Display overall metrics table
            st.subheader("Overall Test Similarity Metrics")
            st.dataframe(overall_df, hide_index=True, use_container_width=True)

            # Provide download option
            csv = overall_df.to_csv(index=False)
            st.download_button(
                label="Download Analysis as CSV",
                data=csv,
                file_name="test_overall_analysis.csv",
                mime="text/csv",
            )

            # Visualizations
            st.subheader("Comparison of Similarity Methods")

            # Create bar chart comparing the methods
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(overall_df))
            width = 0.25

            # Plot bars for each similarity method
            # Define colormaps for each method
            # Normalize values to [0, 1] range for colormap scaling
            tfidf_norm = (overall_df["TF-IDF Similarity"] - overall_df["TF-IDF Similarity"].min()) / (overall_df["TF-IDF Similarity"].max() - overall_df["TF-IDF Similarity"].min())
            bert_norm = (overall_df["BERT Similarity"] - overall_df["BERT Similarity"].min()) / (overall_df["BERT Similarity"].max() - overall_df["BERT Similarity"].min())
            rouge_norm = (overall_df["ROUGE-L Score"] - overall_df["ROUGE-L Score"].min()) / (overall_df["ROUGE-L Score"].max() - overall_df["ROUGE-L Score"].min())

            # Generate colors based on normalized values
            tfidf_colors = plt.cm.Blues(tfidf_norm)
            bert_colors = plt.cm.Greens(bert_norm)
            rouge_colors = plt.cm.Reds(rouge_norm)

            # Plot bars for each similarity method with colors
            bars_tfidf = ax.bar(x - width, overall_df["TF-IDF Similarity"], width, label='TF-IDF', color=tfidf_colors)
            bars_bert = ax.bar(x, overall_df["BERT Similarity"], width, label='BERT', color=bert_colors)
            bars_rouge = ax.bar(x + width, overall_df["ROUGE-L Score"], width, label='ROUGE-L', color=rouge_colors)

            # Add text on each bar with the value
            for bar in bars_tfidf:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', 
                        ha='center', va='bottom', fontsize=8)
            for bar in bars_bert:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', 
                        ha='center', va='bottom', fontsize=8)
            for bar in bars_rouge:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', 
                        ha='center', va='bottom', fontsize=8)

            # Add labels and legend
            ax.set_xlabel('Test Name')
            ax.set_ylabel('Similarity Score')
            ax.set_title('Similarity Scores by Method')
            ax.set_xticks(x)
            ax.set_xticklabels(overall_df["Test Name"], rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1.05)

            plt.tight_layout()
            st.pyplot(fig)

            # Summary metrics
            st.subheader("Summary Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Average TF-IDF Similarity",
                    value=f"{overall_df['TF-IDF Similarity'].mean():.4f}"
                )

            with col2:
                st.metric(
                    label="Average BERT Similarity",
                    value=f"{overall_df['BERT Similarity'].mean():.4f}"
                )

            with col3:
                st.metric(
                    label="Average ROUGE-L Score",
                    value=f"{overall_df['ROUGE-L Score'].mean():.4f}"
                )

with tab2:
    st.header("Row-by-Row Analysis")
    st.write("Select a specific test to see detailed row-by-row metrics")

    uploaded_file_tab2 = st.file_uploader("Upload the 'tables.html' file", type=["html"], key="row_uploader")

    if uploaded_file_tab2:
        # Save uploaded file temporarily
        temp_path = Path("uploaded_tables_tab2.html")
        temp_path.write_bytes(uploaded_file_tab2.read())

        # Extract test data
        test_data = extract_test_data_from_html(temp_path)

        # Select a test to analyze
        selected_test = st.selectbox("Select a Test for Detailed Analysis",
                                   list(test_data.keys()),
                                   key="detailed_test_selector")

        if selected_test and st.button("Generate Row-by-Row Analysis"):
            # Create progress indicators
            row_analysis_spinner = st.empty()
            row_progress_placeholder = st.empty()

            with row_analysis_spinner.container():
                with st.spinner(f"Analyzing test '{selected_test}' row by row..."):
                    # Create a progress bar
                    progress_bar = row_progress_placeholder.progress(0)

                    # Perform detailed row-by-row analysis
                    result = perform_row_by_row_analysis(selected_test, test_data, row_progress_placeholder)

                    # Clear progress bar
                    row_progress_placeholder.empty()

            if result:
                type_a = result["type_a"]
                type_b = result["type_b"]
                row_analysis_df = result["dataframe"]

                st.success(f"✅ Row-by-row analysis completed for '{selected_test}'")

                # Display test types being compared
                st.write(f"### Comparing Test Types: '{type_a}' vs '{type_b}'")

                # Display the detailed results table
                st.dataframe(row_analysis_df, hide_index=True, use_container_width=True)

                # Provide download option for detailed results
                detailed_csv = row_analysis_df.to_csv(index=False)
                st.download_button(
                    label=f"Download Row-by-Row Analysis for '{selected_test}' as CSV",
                    data=detailed_csv,
                    file_name=f"test_{selected_test}_row_analysis.csv",
                    mime="text/csv",
                )

                # Create visualizations for row metrics
                st.subheader("Row-by-Row Visualizations")

                # Similarity metrics by row
                fig, ax = plt.subplots(figsize=(12, 6))
                x = row_analysis_df["Row #"]

                ax.plot(x, row_analysis_df["TF-IDF Combined"], marker='o', label='TF-IDF Combined')
                ax.plot(x, row_analysis_df["BERT Combined"], marker='s', label='BERT Combined')
                ax.plot(x, row_analysis_df["ROUGE-L"], marker='^', label='ROUGE-L')

                ax.set_xlabel('Row Number')
                ax.set_ylabel('Similarity Score')
                ax.set_title(f'Similarity Metrics by Row for Test: {selected_test}')
                ax.set_xticks(x)
                ax.set_ylim(0, 1.05)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()

                plt.tight_layout()
                st.pyplot(fig)

                # Summary stats for this test
                st.subheader("Summary Statistics")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="Average TF-IDF Combined",
                        value=f"{row_analysis_df['TF-IDF Combined'].mean():.4f}"
                    )

                with col2:
                    st.metric(
                        label="Average BERT Combined",
                        value=f"{row_analysis_df['BERT Combined'].mean():.4f}"
                    )

                with col3:
                    st.metric(
                        label="Average ROUGE-L Score",
                        value=f"{row_analysis_df['ROUGE-L'].mean():.4f}"
                    )

                # Identify rows with notable differences
                st.subheader("Notable Rows")

                # Most similar rows by different metrics
                tfidf_max_idx = row_analysis_df["TF-IDF Combined"].idxmax()
                bert_max_idx = row_analysis_df["BERT Combined"].idxmax()
                rouge_max_idx = row_analysis_df["ROUGE-L"].idxmax()

                # Least similar rows by different metrics
                tfidf_min_idx = row_analysis_df["TF-IDF Combined"].idxmin()
                bert_min_idx = row_analysis_df["BERT Combined"].idxmin()
                rouge_min_idx = row_analysis_df["ROUGE-L"].idxmin()

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**TF-IDF Analysis:**")
                    st.write(f"Most Similar: Row #{row_analysis_df.iloc[tfidf_max_idx]['Row #']} ({row_analysis_df.iloc[tfidf_max_idx]['TF-IDF Combined']:.4f})")
                    st.write(f"Least Similar: Row #{row_analysis_df.iloc[tfidf_min_idx]['Row #']} ({row_analysis_df.iloc[tfidf_min_idx]['TF-IDF Combined']:.4f})")

                with col2:
                    st.write("**BERT Analysis:**")
                    st.write(f"Most Similar: Row #{row_analysis_df.iloc[bert_max_idx]['Row #']} ({row_analysis_df.iloc[bert_max_idx]['BERT Combined']:.4f})")
                    st.write(f"Least Similar: Row #{row_analysis_df.iloc[bert_min_idx]['Row #']} ({row_analysis_df.iloc[bert_min_idx]['BERT Combined']:.4f})")

                with col3:
                    st.write("**ROUGE-L Analysis:**")
                    st.write(f"Most Similar: Row #{row_analysis_df.iloc[rouge_max_idx]['Row #']} ({row_analysis_df.iloc[rouge_max_idx]['ROUGE-L']:.4f})")
                    st.write(f"Least Similar: Row #{row_analysis_df.iloc[rouge_min_idx]['Row #']} ({row_analysis_df.iloc[rouge_min_idx]['ROUGE-L']:.4f})")

            else:
                st.error("Could not perform row-by-row analysis for the selected test.")

with tab3:
    st.header("About This Tool")

    st.markdown("""
    ## Test Similarity Analysis Tool

    This tool provides comprehensive comparison of test similarity using three different methods:

    ### Similarity Methods

    1. **TF-IDF (Term Frequency-Inverse Document Frequency)**
       - Based on word frequency and importance
       - Good for detecting similar vocabulary and terminology
       - Less sensitive to semantics/meaning

    2. **BERT (Bidirectional Encoder Representations from Transformers)**
       - Uses deep learning to understand semantic meaning
       - Can detect similar concepts even when using different wording
       - More computationally intensive but generally more accurate

    3. **ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence)**
       - Measures similarity based on longest common subsequence
       - Good for detecting similar sentence structures and word order
       - Used in text summarization evaluation

    ### How to Use

    1. Upload your `tables.html` file containing the test data
    2. Use the "Overall Analysis" tab to see a summary of all tests
    3. Use the "Row-by-Row Analysis" tab to select a specific test and see detailed metrics for each row

    ### Technical Details

    - BERT model: `paraphrase-MiniLM-L6-v2` (optimized for semantic textual similarity)
    - All metrics are normalized to a scale of 0.0 (no similarity) to 1.0 (identical)
    - Word count is determined using NLTK's word tokenizer

    ### Interpretation Guide

    - **TF-IDF**: Focuses on vocabulary overlap, may miss semantic similarity
    - **BERT**: Best for understanding meaning regardless of exact wording
    - **ROUGE-L**: Good for detecting structural similarities

    For best results, consider all three metrics together to get a complete picture of test similarity.
    """)

# Run the application
if __name__ == "__main__":
    pass