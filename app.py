# flake8: noqa
import datetime
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from rouge import Rouge
from bs4 import BeautifulSoup

# Download nltk data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)

download_nltk_data()
# analysis_df = pd.DataFrame()


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
    vectorizer = TfidfVectorizer().fit([text1, text2])
    vectors = vectorizer.transform([text1, text2])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

# Semantic similarity using Sentence-BERT
def semantic_similarity_bert(text1, text2):
    """Calculate semantic similarity with null value handling."""
    # First check for null values
    null_result = handle_null_values(text1, text2)
    if null_result is not None:
        return null_result

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode([str(text1), str(text2)])
    return np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))

def perform_comparison(df_a, df_b, progress_placeholder):
    """Perform row-wise comparison with progress tracking."""
    row_similarities = []
    # Use the shorter DataFrame length to avoid index errors
    total_rows = min(len(df_a), len(df_b))

    # Create a progress bar
    progress_bar = progress_placeholder.progress(0)

    # Only compare up to the length of the shorter DataFrame
    for i in range(total_rows):
        row_a = df_a.iloc[i]
        row_b = df_b.iloc[i]

        action_a = row_a["Actions"]
        verification_a = row_a["Verifications"]
        action_b = row_b["Actions"]
        verification_b = row_b["Verifications"]

        # Calculate semantic similarity using sentence transformer
        action_similarity = semantic_similarity_bert(action_a, action_b)
        verification_similarity = semantic_similarity_bert(verification_a, verification_b)

        # Average of action and verification similarity as row-wise similarity
        row_similarity = (action_similarity + verification_similarity) / 2
        row_similarities.append(row_similarity)

        # Update progress
        progress = (i + 1) / total_rows
        progress_bar.progress(progress)

    # Clean up progress bar after completion
    progress_placeholder.empty()

    return row_similarities, total_rows

def calculate_rouge_l(text1, text2):
    """Calculate ROUGE-L score between two texts."""
    # Handle null values
    null_result = handle_null_values(text1, text2)
    if null_result is not None:
        return null_result

    # Initialize Rouge
    rouge = Rouge()

    # Rouge expects non-empty strings
    if not str(text1).strip() or not str(text2).strip():
        return 0.0

    try:
        # Calculate ROUGE scores
        scores = rouge.get_scores(str(text1), str(text2))
        return scores[0]['rouge-l']['f']
    except Exception:
        # If there's an error in ROUGE calculation, return 0
        return 0.0

def count_words(text):
    """Count the number of words in a text."""
    if pd.isna(text) or text == "":
        return 0
    return len(word_tokenize(str(text)))

def get_word_count_stats(df):
    """Calculate average word count for Actions and Verifications columns."""
    action_word_counts = df["Actions"].apply(count_words)
    verification_word_counts = df["Verifications"].apply(count_words)

    # Calculate average words per row (Actions + Verifications)
    total_words = action_word_counts.sum() + verification_word_counts.sum()
    if len(df) > 0:
        avg_words_per_row = total_words / len(df)
    else:
        avg_words_per_row = 0

    return avg_words_per_row

def perform_quantitative_analysis(test_data):
    """Perform quantitative analysis for all tests."""
    analysis_results = []

    for test_name, types in test_data.items():
        if len(types) == 2:
            type_a = list(types.keys())[0]
            type_b = list(types.keys())[1]

            df_a = types[type_a]
            df_b = types[type_b]

            # Calculate metrics
            rows_a = len(df_a)
            rows_b = len(df_b)
            avg_words_a = get_word_count_stats(df_a)
            avg_words_b = get_word_count_stats(df_b)
            diff_avg_words = abs(avg_words_a - avg_words_b)

            # Calculate ROUGE-L
            rouge_scores = []
            for i, row_a in df_a.iterrows():
                if i < len(df_b):
                    row_b = df_b.iloc[i]

                    # Combine Actions and Verifications for ROUGE comparison
                    text_a = str(row_a["Actions"]) + " " + str(row_a["Verifications"])
                    text_b = str(row_b["Actions"]) + " " + str(row_b["Verifications"])

                    rouge_score = calculate_rouge_l(text_a, text_b)
                    rouge_scores.append(rouge_score)

            avg_rouge_l = np.mean(rouge_scores) if rouge_scores else 0

            # Add to results
            analysis_results.append({
                "Test Name": test_name,
                "Test A": type_a,
                "Test B": type_b,
                "Rows A": rows_a,
                "Rows B": rows_b,
                "Avg Words/Row A": round(avg_words_a, 2),
                "Avg Words/Row B": round(avg_words_b, 2),
                "Diff Avg Words": round(diff_avg_words, 2),
                "ROUGE-L": round(avg_rouge_l, 4)
            })

    return pd.DataFrame(analysis_results)

def perform_row_by_row_analysis(test_name, test_data, progress_bar=None):
    """Perform row-by-row analysis for a specific test."""
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
    #normalize progress bar steps by rows_to compare length, attributing 100% to the last row
    if rows_to_compare > 0:
        progress_bar_steps = 100 / rows_to_compare
        progess_bar_steps_list = [i * progress_bar_steps/100 for i in range(rows_to_compare)]
    else:
        progress_bar_steps = 0


    for i in range(rows_to_compare):
        row_a = df_a.iloc[i]
        row_b = df_b.iloc[i]
        if progress_bar:
            progress_bar.progress(progess_bar_steps_list[i])

        # Extract text from both rows
        action_a = str(row_a["Actions"])
        verification_a = str(row_a["Verifications"])
        action_b = str(row_b["Actions"])
        verification_b = str(row_b["Verifications"])

        # Combined text for ROUGE-L
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

        # Calculate similarities
        action_similarity = semantic_similarity_bert(action_a, action_b)
        verification_similarity = semantic_similarity_bert(verification_a, verification_b)
        combined_similarity = (action_similarity + verification_similarity) / 2
        rouge_score = calculate_rouge_l(text_a, text_b)

        # Add to results
        row_analysis_results.append({
            "Row #": i + 1,
            "Actions A Words": action_a_words,
            "Actions B Words": action_b_words,
            "Verifications A Words": verification_a_words,
            "Verifications B Words": verification_b_words,
            "Total Words A": total_words_a,
            "Total Words B": total_words_b,
            "Word Difference": word_diff,
            "Actions Similarity": round(action_similarity, 4),
            "Verifications Similarity": round(verification_similarity, 4),
            "Combined Similarity": round(combined_similarity, 4),
            "ROUGE-L": round(rouge_score, 4)
        })

    result_df = pd.DataFrame(row_analysis_results)

    return {
        "type_a": type_a,
        "type_b": type_b,
        "dataframe": result_df
    }


# Streamlit application
st.set_page_config(
    page_title="Test Similarity Comparison",
    page_icon="🧪",
    layout="wide"
)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Main Analysis", "Quantitative Analysis", "About"])

with tab1:
    st.title("Semantic and Cosine Similarity Comparison by Test Name and Type")
    uploaded_file = st.file_uploader("Upload the 'tables.html' file", type=["html"], key="main_uploader")

    if uploaded_file:
        # Save uploaded file temporarily
        temp_path = Path("uploaded_tables.html")
        temp_path.write_bytes(uploaded_file.read())

        # Explanation and Formula before overall similarity computation
        st.subheader("How is the Overall Similarity Calculated?")
        explanation = """
    ### Similarity Score Calculation

    The similarity score you see in the graph is calculated through these steps:

    1. **Row-by-Row Comparison**:
       - For each row, we compare both the 'Actions' and 'Verifications' columns between the two test types
       - We use BERT (Bidirectional Encoder Representations from Transformers) to understand the semantic meaning of the text

    2. **Handling Missing Values**:
       - If both cells are empty (null): Score = 1.0 (100% similar)
       - If one cell is empty and the other isn't: Score = 0.0 (0% similar)
       - If neither cell is empty: Calculate semantic similarity

    3. **Individual Row Score**:
       - Action Similarity: Calculated using BERT (0.0 to 1.0)
       - Verification Similarity: Calculated using BERT (0.0 to 1.0)
       - Row Score = (Action Similarity + Verification Similarity) / 2

    4. **Final Score**:
       - Average of all row scores
       - Displayed as a value between 0.0 (completely different) and 1.0 (identical)

    The color intensity in the graph represents this final score:
    - Darker/more intense colors = Higher similarity
    - Lighter colors = Lower similarity
    """

        st.markdown(explanation)
        # Extract test data
        test_data = extract_test_data_from_html(temp_path)

        if st.button("Generate Overall Similarity Comparison"):
            # Display a graph for the overall similarity
            st.subheader("Overall Similarity Comparison")
            fig, ax = plt.subplots()

            # Create placeholders for loading indicators
            spinner_placeholder = st.empty()
            progress_placeholder = st.empty()
            #calculate elapsed time
            start = datetime.datetime.now()
            with spinner_placeholder.container():
                # Progress Bar initialization for this block
                progress_bar = progress_placeholder.progress(0)
                num_tests = len(test_data)  # Total number of tests
                overall_similarity_scores : list[float] = []

                for idx, (test_name, types) in enumerate(test_data.items()):
                    with st.spinner(f"Processing test '{test_name}'..."):
                        if len(types) == 2:
                            type_a = list(types.keys())[0]
                            type_b = list(types.keys())[1]

                            # Perform semantic comparison for Actions and Verifications
                            type_a_df = types[type_a]
                            type_b_df = types[type_b]

                            similarity_scores = []
                            for i, row_a in type_a_df.iterrows():
                                action_a = row_a["Actions"]
                                verification_a = row_a["Verifications"]
                                if i < len(type_b_df):
                                    row_b = type_b_df.iloc[i]
                                    action_b = row_b["Actions"]
                                    verification_b = row_b["Verifications"]

                                    # Calculate semantic similarity
                                    action_similarity = semantic_similarity_bert(action_a, action_b)
                                    verification_similarity = semantic_similarity_bert(verification_a, verification_b)

                                    # Average of action and verification similarity
                                    row_similarity = (action_similarity + verification_similarity) / 2
                                    similarity_scores.append(row_similarity)

                            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
                            overall_similarity_scores.append(avg_similarity)

                            # Gradients for color intensity
                            color_intensity = avg_similarity  # Use the average similarity directly for color intensity
                            bar_color = plt.cm.viridis(color_intensity)  # Use viridis colormap for gradient colors

                            # Draw the bar with gradient color
                            ax.barh(test_name, avg_similarity, color=bar_color)
                            ax.text(avg_similarity, test_name, f"{avg_similarity:.4f}", va='center', ha='left', color='black')

                        # Update progress bar
                        progress = (idx + 1) / num_tests
                        progress_bar.progress(progress)

                # Clear the loading indicators
                spinner_placeholder.success("✅ Overall comparison completed!")
                progress_placeholder.markdown('')

            with st.container():
                # Calculate elapsed time
                end = datetime.datetime.now()
                elapsed_time = end - start
                formatted_elapsed_time = elapsed_time.total_seconds()
                st.write(f"⏱️ Elapsed Time: {formatted_elapsed_time} seconds")
                plt.xlabel("Average Similarity")
                plt.ylabel("Test Name")
                plt.title("Similarity Comparison by Test Name")
                st.pyplot(fig)

                # Display overall metrics
                a,b,c,d = st.columns(4)
                a.metric(
                    label="Total Tests 📊",
                    value=f"{num_tests}"
                )
                b.metric(
                    label="Average Similarity 📈",
                    value=f"{np.mean(overall_similarity_scores):.4f}",
                )
                c.metric(
                    label="Minimum Similarity 📉",
                    value=f"{min(overall_similarity_scores):.4f}",
                )
                d.metric(
                    label="Maximum Similarity 🏆",
                    value=f"{max(overall_similarity_scores):.4f}",
                )

        st.markdown('---')

        selected_test = st.selectbox("Select a Test Name to Compare", list(test_data.keys()))
        if selected_test:
            available_types = list(test_data[selected_test].keys())
            type_a = available_types[0]
            type_b = available_types[1]

            if st.button("Compare"):
                if type_a in test_data[selected_test] and type_b in test_data[selected_test]:
                    # Create placeholders for loading indicators
                    spinner_placeholder = st.empty()
                    progress_placeholder = st.empty()
                    spinner_container = spinner_placeholder.container()

                    spinner_container.write("🔄 Performing comparison analysis...")

                    # Retrieve DataFrames for display
                    df_a = test_data[selected_test][type_a]
                    df_b = test_data[selected_test][type_b]

                    # Perform comparison with progress tracking
                    with spinner_placeholder.container():
                        with st.spinner("Calculating similarities..."):
                            row_similarities, compared_rows = perform_comparison(df_a, df_b, progress_placeholder)

                    # Calculate average similarity
                    avg_similarity = np.mean(row_similarities) if row_similarities else 0

                    # Clear the loading message
                    spinner_placeholder.success("✅ Comparison analysis completed!")

                        # Display results
                    with st.container():
                        st.write(f"### Comparison Results for Test '{selected_test}'")
                        # Create columns for the metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                label="Average Similarity",
                                value=f"{avg_similarity:.4f}"
                            )
                        with col2:
                            st.metric(
                                label="Minimum Similarity",
                                value=f"{min(row_similarities):.4f}" if row_similarities else "N/A"
                            )
                        with col3:
                            st.metric(
                                label="Maximum Similarity",
                                value=f"{max(row_similarities):.4f}" if row_similarities else "N/A"
                            )
                        with col4:
                            st.metric(
                                label="Rows Compared",
                                value=f"{compared_rows}"
                            )

                        # Display detailed results
                        with st.expander("View Detailed Results", expanded=False):
                            # Display length difference warning if applicable
                            if len(df_a) != len(df_b):
                                st.warning(f"⚠️ The two test types have different numbers of rows: {type_a}: {len(df_a)} rows, {type_b}: {len(df_b)} rows. Only the first {compared_rows} rows were compared.")

                            # Create copies of original DataFrames
                            df_a_display = df_a.copy()
                            df_b_display = df_b.copy()

                            # Initialize similarity columns with 'N/A'
                            df_a_display.loc[:, 'Similarity'] = 'N/A'
                            df_b_display.loc[:, 'Similarity'] = 'N/A'

                            # Update similarity values for compared rows
                            for i, similarity in enumerate(row_similarities):
                                if i < len(df_a_display):
                                    df_a_display.at[df_a_display.index[i], 'Similarity'] = f"{similarity:.4f}"
                                if i < len(df_b_display):
                                    df_b_display.at[df_b_display.index[i], 'Similarity'] = f"{similarity:.4f}"

                            st.write(f"**Type '{type_a}' Table Data:**")
                            st.dataframe(df_a_display, hide_index=True)

                            st.write(f"**Type '{type_b}' Table Data:**")
                            st.dataframe(df_b_display, hide_index=True)

                else:
                    st.error(f"One or both types ('{type_a}', '{type_b}') are missing for the selected test.")

with tab2:
    st.title("Quantitative Analysis")

    uploaded_file_tab2 = st.file_uploader("Upload the 'tables.html' file", type=["html"], key="quant_uploader")

    if uploaded_file_tab2:
        # Save uploaded file temporarily
        temp_path = Path("uploaded_tables_tab2.html")
        temp_path.write_bytes(uploaded_file_tab2.read())

        # Extract test data
        test_data = extract_test_data_from_html(temp_path)

        # Create tabs for different types of analysis
        quant_tab1, quant_tab2 = st.tabs(["Overall Test Metrics", "Row-by-Row Analysis"])

        with quant_tab1:
            if st.button("Generate Overall Quantitative Analysis", key="overall_analysis"):
                st.subheader("Quantitative Metrics by Test")

                # Create placeholder for progress
                analysis_spinner = st.empty()

                with analysis_spinner.container():
                    with st.spinner("Calculating quantitative metrics..."):
                        # Perform quantitative analysis
                        analysis_df = perform_quantitative_analysis(test_data)
                    # Additional analysis - test with most differences
                    st.subheader("Key Insights")

                    # Test with highest word count difference
                    max_word_diff_idx = analysis_df["Diff Avg Words"].idxmax()
                    max_word_diff_test = analysis_df.iloc[max_word_diff_idx]

                    # Test with lowest ROUGE-L score (most different)
                    min_rouge_idx = analysis_df["ROUGE-L"].idxmin()
                    min_rouge_test = analysis_df.iloc[min_rouge_idx]

                    # Test with highest ROUGE-L score (most similar)
                    max_rouge_idx = analysis_df["ROUGE-L"].idxmax()
                    max_rouge_test = analysis_df.iloc[max_rouge_idx]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            label="Highest Word Count Difference",
                            value=max_word_diff_test["Test Name"],
                            delta=f"{max_word_diff_test['Diff Avg Words']:.2f} words"
                        )

                    with col2:
                        st.metric(
                            label="Most Different Tests (ROUGE-L)",
                            value=min_rouge_test["Test Name"],
                            delta=f"{min_rouge_test['ROUGE-L']:.4f}",
                            delta_color="inverse"
                        )

                    with col3:
                        st.metric(
                            label="Most Similar Tests (ROUGE-L)",
                            value=max_rouge_test["Test Name"],
                            delta=f"{max_rouge_test['ROUGE-L']:.4f}"
                        )

                # Display the analysis results table
                st.dataframe(analysis_df, hide_index=True, use_container_width=True)

                # Provide download option
                csv = analysis_df.to_csv(index=False)
                st.download_button(
                    label="Download Analysis as CSV",
                    data=csv,
                    file_name="test_quantitative_analysis.csv",
                    mime="text/csv",
                )

                if not analysis_df.empty:
                    st.subheader("Visualizations")

                    # Create tabs for different visualizations
                    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Row Counts", "Word Counts", "ROUGE-L Scores"])

                    with viz_tab1:
                        # Row count comparison
                        fig1, ax1 = plt.subplots(figsize=(10, 6))
                        x = np.arange(len(analysis_df))
                        width = 0.35

                        ax1.bar(x - width/2, analysis_df["Rows A"], width, label=f'Test A Rows')
                        ax1.bar(x + width/2, analysis_df["Rows B"], width, label=f'Test B Rows')

                        ax1.set_xlabel('Test Name')
                        ax1.set_ylabel('Number of Rows')
                        ax1.set_title('Row Count Comparison')
                        ax1.set_xticks(x)
                        ax1.set_xticklabels(analysis_df["Test Name"], rotation=45, ha='right')
                        ax1.legend()

                        plt.tight_layout()
                        st.pyplot(fig1)

                    with viz_tab2:
                        # Word count comparison
                        fig2, ax2 = plt.subplots(figsize=(10, 6))

                        ax2.bar(x - width/2, analysis_df["Avg Words/Row A"], width, label=f'Test A Avg Words/Row')
                        ax2.bar(x + width/2, analysis_df["Avg Words/Row B"], width, label=f'Test B Avg Words/Row')

                        ax2.set_xlabel('Test Name')
                        ax2.set_ylabel('Average Words per Row')
                        ax2.set_title('Word Count Comparison')
                        ax2.set_xticks(x)
                        ax2.set_xticklabels(analysis_df["Test Name"], rotation=45, ha='right')
                        ax2.legend()

                        plt.tight_layout()
                        st.pyplot(fig2)

                    with viz_tab3:
                        # ROUGE-L scores
                        fig3, ax3 = plt.subplots(figsize=(10, 6))
                        norm = plt.Normalize(analysis_df["ROUGE-L"].min(), analysis_df["ROUGE-L"].max())
                        colors = plt.cm.viridis(norm(analysis_df["ROUGE-L"]))

                        ax3.bar(analysis_df["Test Name"], analysis_df["ROUGE-L"], color=colors)

                        ax3.set_xlabel('Test Name')
                        ax3.set_ylabel('ROUGE-L Score')
                        ax3.set_title('ROUGE-L Scores by Test')
                        plt.xticks(rotation=45, ha='right')

                        for i, value in enumerate(analysis_df["ROUGE-L"]):
                            ax3.text(i, value, f"{value:.4f}", ha='center', va='bottom')

                        plt.tight_layout()
                        st.pyplot(fig3)


        with quant_tab2:
            st.subheader("Row-by-Row Analysis")

            # Select a test to analyze in detail
            selected_test = st.selectbox("Select a Test for Detailed Analysis",
                                         list(test_data.keys()),
                                         key="detailed_test_selector")

            if selected_test and st.button("Generate Row-by-Row Analysis", key="row_analysis"):
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
                    fig1, ax1 = plt.subplots(figsize=(12, 6))
                    x = row_analysis_df["Row #"]

                    ax1.plot(x, row_analysis_df["Actions Similarity"], marker='o', label='Actions Similarity')
                    ax1.plot(x, row_analysis_df["Verifications Similarity"], marker='s', label='Verifications Similarity')
                    ax1.plot(x, row_analysis_df["Combined Similarity"], marker='^', label='Combined Similarity')
                    ax1.plot(x, row_analysis_df["ROUGE-L"], marker='d', label='ROUGE-L')

                    ax1.set_xlabel('Row Number')
                    ax1.set_ylabel('Similarity Score')
                    ax1.set_title(f'Similarity Metrics by Row for Test: {selected_test}')
                    ax1.set_xticks(x)
                    ax1.set_ylim(0, 1.05)
                    ax1.grid(True, linestyle='--', alpha=0.7)
                    ax1.legend()

                    plt.tight_layout()
                    st.pyplot(fig1)

                    # Word count comparison by row
                    # fig2, ax2 = plt.subplots(figsize=(12, 6))

                    # ax2.bar(x - 0.2, row_analysis_df["Total Words A"], width=0.4, label=f'Type {type_a} Words')
                    # ax2.bar(x + 0.2, row_analysis_df["Total Words B"], width=0.4, label=f'Type {type_b} Words')

                    # # Add word difference as text
                    # for i, row in row_analysis_df.iterrows():
                    #     row_num = row["Row #"]
                    #     word_diff = row["Word Difference"]
                    #     max_words = max(row["Total Words A"], row["Total Words B"])
                    #     ax2.text(row_num, max_words + 2, f"Δ{word_diff}", ha='center')

                    # ax2.set_xlabel('Row Number')
                    # ax2.set_ylabel('Word Count')
                    # ax2.set_title(f'Word Count Comparison by Row for Test: {selected_test}')
                    # ax2.set_xticks(x)
                    # ax2.legend()

                    # plt.tight_layout()
                    # st.pyplot(fig2)

                    # Summary stats for this test
                    st.subheader("Summary Statistics")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            label="Average Combined Similarity",
                            value=f"{row_analysis_df['Combined Similarity'].mean():.4f}"
                        )

                    with col2:
                        st.metric(
                            label="Average ROUGE-L Score",
                            value=f"{row_analysis_df['ROUGE-L'].mean():.4f}"
                        )

                    with col3:
                        st.metric(
                            label="Average Word Difference per Row",
                            value=f"{row_analysis_df['Word Difference'].mean():.1f}"
                        )

                    # Identify rows with notable differences
                    st.subheader("Notable Rows")

                    # Most similar row
                    most_similar_idx = row_analysis_df["Combined Similarity"].idxmax()
                    most_similar_row = row_analysis_df.iloc[most_similar_idx]

                    # Least similar row
                    least_similar_idx = row_analysis_df["Combined Similarity"].idxmin()
                    least_similar_row = row_analysis_df.iloc[least_similar_idx]

                    # Row with biggest word difference
                    biggest_diff_idx = row_analysis_df["Word Difference"].idxmax()
                    biggest_diff_row = row_analysis_df.iloc[biggest_diff_idx]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write("**Most Similar Row:**")
                        st.write(f"Row #{most_similar_row['Row #']}")
                        st.write(f"Similarity: {most_similar_row['Combined Similarity']:.4f}")

                    with col2:
                        st.write("**Least Similar Row:**")
                        st.write(f"Row #{least_similar_row['Row #']}")
                        st.write(f"Similarity: {least_similar_row['Combined Similarity']:.4f}")

                    with col3:
                        st.write("**Biggest Word Difference:**")
                        st.write(f"Row #{biggest_diff_row['Row #']}")
                        st.write(f"Difference: {biggest_diff_row['Word Difference']} words")

                else:
                    st.error("Could not perform row-by-row analysis for the selected test.")
            

            # Show some visualizations for the analysis
            

with tab3:
    st.title("About This Tool")

    st.markdown("""
    ## Test Similarity Analysis Tool

    This tool provides semantic and quantitative analysis for test comparisons. Use it to:

    1. **Compare test similarity** using semantic analysis with BERT transformer models
    2. **Analyze quantitative metrics** including:
       - Row counts for test A and B
       - Average word counts per row
       - Word count differences
       - ROUGE-L scores for text similarity

    ### Key Features

    - **Main Analysis Tab**: Compare test similarity using semantic understanding
    - **Quantitative Analysis Tab**: Get detailed metrics and statistical comparisons
    - **Visualizations**: View graphical representations of your test comparisons

    ### How to Use

    1. Upload your `tables.html` file containing the test data
    2. Use the different tabs to analyze your tests in various ways
    3. Download CSV reports for further analysis

    ### Metrics Explained

    - **ROUGE-L**: A measure of the longest common subsequence between two texts
    - **Semantic Similarity**: Using BERT to understand the meaning of text beyond simple word matching
    - **Word Count Analysis**: Comparing verbosity between test types

    ### Need Help?

    Contact the developer for assistance or feature requests.
    """)