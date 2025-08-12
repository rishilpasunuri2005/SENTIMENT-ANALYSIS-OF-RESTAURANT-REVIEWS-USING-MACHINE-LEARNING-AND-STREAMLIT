import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize global variables
model = None
vectorizer = None
lemmatizer = None
stop_words = None


def preprocess_text(text):
    """Preprocess review text: lowercase, remove special chars, remove stopwords, lemmatize"""
    global lemmatizer, stop_words

    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', '', str(text).lower())

    # Tokenize and remove stopwords
    tokens = [word for word in text.split() if word not in stop_words]

    # Lemmatize tokens
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(lemmatized)


def train_model(data_file):
    """Train sentiment analysis model on provided data"""
    global model, vectorizer, lemmatizer, stop_words

    # Initialize NLTK resources
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Load and prepare data
    df = pd.read_csv(data_file)

    # Ensure the required columns exist
    if 'review_text' not in df.columns or 'rating' not in df.columns:
        raise ValueError("Dataset must contain 'review_text' and 'rating' columns")

    # Convert ratings to sentiment (1-2: negative, 3: neutral, 4-5: positive)
    df['sentiment'] = df['rating'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))

    # Preprocess review text
    df['processed_text'] = df['review_text'].apply(preprocess_text)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['sentiment'], test_size=0.2, random_state=42
    )

    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(C=1.0, solver='liblinear', max_iter=200, multi_class='ovr')
    model.fit(X_train_tfidf, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'])
    cm = confusion_matrix(y_test, y_pred)

    # Save model and vectorizer
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, 'models/sentiment_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'model_path': 'models/sentiment_model.pkl',
        'vectorizer_path': 'models/tfidf_vectorizer.pkl'
    }


def load_model():
    """Load trained model and vectorizer"""
    global model, vectorizer, lemmatizer, stop_words

    # Initialize NLTK resources if not already loaded
    if lemmatizer is None:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

    # Load model and vectorizer if they exist
    if os.path.exists('models/sentiment_model.pkl') and os.path.exists('models/tfidf_vectorizer.pkl'):
        model = joblib.load('models/sentiment_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return True
    return False


def analyze_sentiment(review_text):
    """Predict sentiment for a given review text"""
    global model, vectorizer

    if model is None or vectorizer is None:
        if not load_model():
            return {
                'error': 'Model not trained yet. Please train the model first.'
            }

    # Preprocess input text
    processed_text = preprocess_text(review_text)

    # Transform text to TF-IDF features
    text_tfidf = vectorizer.transform([processed_text])

    # Predict sentiment
    sentiment_id = model.predict(text_tfidf)[0]
    sentiment_proba = model.predict_proba(text_tfidf)[0]

    # Map sentiment ID to label
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = sentiment_map[sentiment_id]

    # Confidence score (probability of predicted class)
    confidence = sentiment_proba[sentiment_id]

    return {
        'sentiment': sentiment,
        'confidence': float(confidence),
        'probabilities': {
            'Negative': float(sentiment_proba[0]),
            'Neutral': float(sentiment_proba[1]),
            'Positive': float(sentiment_proba[2])
        }
    }


def batch_analyze(df):
    """Analyze multiple reviews from a dataframe"""
    if 'review_text' not in df.columns:
        return {'error': 'CSV must contain a "review_text" column'}

    # Analyze each review
    results = []
    for review in df['review_text']:
        result = analyze_sentiment(review)
        if 'error' in result:
            return result
        results.append(result)

    # Add results to dataframe
    df['sentiment'] = [r['sentiment'] for r in results]
    df['confidence'] = [r['confidence'] for r in results]

    return {'results': results, 'dataframe': df}


def main():
    st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

    st.title("Sentiment Analysis Application")
    st.markdown("This app analyzes the sentiment of reviews using machine learning.")

    # Initialize necessary directories
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Load model if it exists
    model_loaded = load_model()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Single Review Analysis", "Batch Analysis", "Train Model"])

    if page == "Single Review Analysis":
        st.header("Analyze a Single Review")

        if not model_loaded:
            st.warning("Model not trained yet. Please go to the 'Train Model' page to train a model first.")
        else:
            review_text = st.text_area("Enter a review to analyze:", height=150)

            if st.button("Analyze Sentiment"):
                if review_text:
                    with st.spinner("Analyzing sentiment..."):
                        result = analyze_sentiment(review_text)

                        if 'error' in result:
                            st.error(result['error'])
                        else:
                            # Display result with color coding
                            sentiment = result['sentiment']
                            confidence = result['confidence']

                            col1, col2 = st.columns(2)

                            with col1:
                                if sentiment == 'Positive':
                                    st.markdown(f"<h3 style='color:green'>Sentiment: {sentiment}</h3>",
                                                unsafe_allow_html=True)
                                elif sentiment == 'Negative':
                                    st.markdown(f"<h3 style='color:red'>Sentiment: {sentiment}</h3>",
                                                unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<h3 style='color:orange'>Sentiment: {sentiment}</h3>",
                                                unsafe_allow_html=True)

                                st.markdown(f"<h4>Confidence: {confidence:.2%}</h4>", unsafe_allow_html=True)

                            with col2:
                                # Create a horizontal bar chart for probabilities
                                probabilities = result['probabilities']
                                fig, ax = plt.subplots(figsize=(8, 3))
                                sentiment_labels = list(probabilities.keys())
                                sentiment_values = list(probabilities.values())

                                colors = ['red', 'orange', 'green']
                                bars = ax.barh(sentiment_labels, sentiment_values, color=colors)

                                # Add value labels
                                for bar in bars:
                                    width = bar.get_width()
                                    var = 47
                                    ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.2%}',
                                            ha='left', va='center')

                                ax.set_xlim(0, 1)
                                ax.set_xlabel('Probability')
                                ax.set_title('Sentiment Probabilities')

                                st.pyplot(fig)
                else:
                    st.warning("Please enter a review to analyze.")

    elif page == "Batch Analysis":
        st.header("Batch Analysis")

        if not model_loaded:
            st.warning("Model not trained yet. Please go to the 'Train Model' page to train a model first.")
        else:
            st.write("Upload a CSV file with a 'review_text' column to perform batch sentiment analysis.")

            # Add sample format explanation
            with st.expander("See expected CSV format"):
                st.write(
                    "Your CSV should contain at least a 'review_text' column. Additional columns will be preserved in the results.")
                st.code(
                    "review_text,other_column\n\"This product is amazing!\",data1\n\"I didn't like this service.\",data2")

            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)

                    if 'review_text' not in df.columns:
                        st.error("CSV must contain a 'review_text' column")
                    else:
                        # Show data preview with option to limit rows
                        st.subheader("Data Preview")
                        num_preview_rows = min(5, len(df))
                        st.dataframe(df.head(num_preview_rows))
                        st.write(f"Loaded {len(df)} reviews.")

                        # Add analysis options
                        with st.expander("Analysis Options"):
                            # Set the full dataset as default
                            use_full_dataset = st.checkbox("Analyze full dataset", value=True)

                            if not use_full_dataset:
                                sample_size = st.slider(
                                    "Sample size",
                                    min_value=10,
                                    max_value=len(df),
                                    value=min(len(df), 1000),
                                    step=10
                                )
                            else:
                                sample_size = len(df)

                            # Add option for batch processing to handle large datasets more efficiently
                            batch_size = st.slider(
                                "Processing batch size",
                                min_value=100,
                                max_value=5000,
                                value=500,
                                step=100,
                                help="Larger batch sizes process faster but may use more memory"
                            )

                        if st.button("Analyze Batch"):
                            # Use a sample if needed
                            analysis_df = df if use_full_dataset else df.sample(sample_size, random_state=42)

                            # Create a progress bar
                            progress_bar = st.progress(0)
                            progress_text = st.empty()
                            status_text = st.empty()

                            with st.spinner("Analyzing batch of reviews..."):
                                # Process reviews in batches with progress updates
                                results = []
                                total_reviews = len(analysis_df)

                                status_text.text(f"Starting analysis of {total_reviews} reviews...")

                                # Process in batches for better performance with large datasets
                                for i in range(0, total_reviews, batch_size):
                                    batch_end = min(i + batch_size, total_reviews)
                                    batch = analysis_df.iloc[i:batch_end]

                                    status_text.text(
                                        f"Processing batch {i // batch_size + 1} of {(total_reviews - 1) // batch_size + 1}...")

                                    # Process each review in the current batch
                                    batch_results = []
                                    for j, review in enumerate(batch['review_text']):
                                        # Update progress
                                        progress = (i + j + 1) / total_reviews
                                        progress_bar.progress(progress)
                                        progress_text.text(
                                            f"Processing review {i + j + 1}/{total_reviews} ({progress:.1%})")

                                        # Analyze sentiment
                                        result = analyze_sentiment(review)
                                        if 'error' in result:
                                            st.error(result['error'])
                                            break
                                        batch_results.append(result)

                                    # If error occurred, break out of the loop
                                    if 'error' in batch_results:
                                        break

                                    # Add batch results to overall results
                                    results.extend(batch_results)

                                # Clear progress indicators when done
                                progress_text.empty()
                                status_text.text("Analysis complete!")

                                if len(results) == len(analysis_df):
                                    # Add results to dataframe
                                    results_df = analysis_df.copy()
                                    results_df['sentiment'] = [r['sentiment'] for r in results]
                                    results_df['confidence'] = [r['confidence'] for r in results]
                                    results_df['negative_prob'] = [r['probabilities']['Negative'] for r in results]
                                    results_df['neutral_prob'] = [r['probabilities']['Neutral'] for r in results]
                                    results_df['positive_prob'] = [r['probabilities']['Positive'] for r in results]

                                    # Display results in tabs
                                    tab1, tab2, tab3 = st.tabs(
                                        ["Results Table", "Visualizations", "Summary Statistics"])

                                    with tab1:
                                        st.subheader("Analysis Results")

                                        # Add filtering options
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            sentiment_filter = st.multiselect(
                                                "Filter by sentiment",
                                                options=["Positive", "Neutral", "Negative"],
                                                default=["Positive", "Neutral", "Negative"]
                                            )

                                        with col2:
                                            min_confidence = st.slider(
                                                "Minimum confidence score",
                                                min_value=0.0,
                                                max_value=1.0,
                                                value=0.0,
                                                step=0.05
                                            )

                                        # Apply filters
                                        filtered_df = results_df[
                                            (results_df['sentiment'].isin(sentiment_filter)) &
                                            (results_df['confidence'] >= min_confidence)
                                            ]

                                        st.write(f"Showing {len(filtered_df)} of {len(results_df)} results")

                                        # For large datasets, limit the display (but not the data)
                                        if len(filtered_df) > 1000:
                                            st.write(
                                                "Displaying first 1000 rows (all rows will be included in downloads)")
                                            st.dataframe(filtered_df.head(1000))
                                        else:
                                            st.dataframe(filtered_df)

                                    with tab2:
                                        st.subheader("Visualizations")

                                        col1, col2 = st.columns(2)

                                        with col1:
                                            # Sentiment distribution chart
                                            st.write("Sentiment Distribution")
                                            fig, ax = plt.subplots(figsize=(8, 5))
                                            sentiment_counts = results_df['sentiment'].value_counts()

                                            # Ensure all three sentiments are represented in the plot
                                            for sentiment in ['Negative', 'Neutral', 'Positive']:
                                                if sentiment not in sentiment_counts:
                                                    sentiment_counts[sentiment] = 0

                                            sentiment_counts = sentiment_counts.reindex(
                                                ['Negative', 'Neutral', 'Positive'])

                                            # Create bar chart with percentage labels
                                            total = sentiment_counts.sum()
                                            bars = ax.bar(
                                                sentiment_counts.index,
                                                sentiment_counts.values,
                                                color=['red', 'orange', 'green']
                                            )

                                            # Add percentage labels
                                            for bar in bars:
                                                height = bar.get_height()
                                                percentage = height / total * 100 if total > 0 else 0
                                                ax.text(
                                                    bar.get_x() + bar.get_width() / 2.,
                                                    height + 0.1,
                                                    f'{height} ({percentage:.1f}%)',
                                                    ha='center', va='bottom'
                                                )

                                            ax.set_ylim(0, max(sentiment_counts.values) * 1.2 if total > 0 else 1)
                                            ax.set_xlabel('Sentiment')
                                            ax.set_ylabel('Count')
                                            st.pyplot(fig)

                                        with col2:
                                            # Confidence distribution chart
                                            st.write("Confidence Score Distribution")
                                            fig, ax = plt.subplots(figsize=(8, 5))

                                            # Create histograms by sentiment
                                            for sentiment, color in zip(['Positive', 'Neutral', 'Negative'],
                                                                        ['green', 'orange', 'red']):
                                                subset = results_df[results_df['sentiment'] == sentiment]
                                                if len(subset) > 0:
                                                    sns.histplot(
                                                        subset['confidence'],
                                                        bins=10,
                                                        alpha=0.6,
                                                        label=sentiment,
                                                        color=color,
                                                        ax=ax
                                                    )

                                            ax.set_xlabel('Confidence Score')
                                            ax.set_ylabel('Frequency')
                                            ax.legend()
                                            st.pyplot(fig)

                                        # Probability distribution chart
                                        st.write("Probability Distribution Across All Reviews")
                                        fig, ax = plt.subplots(figsize=(10, 5))

                                        # Prepare data for boxplot
                                        prob_data = {
                                            'Negative': results_df['negative_prob'],
                                            'Neutral': results_df['neutral_prob'],
                                            'Positive': results_df['positive_prob']
                                        }

                                        # Create boxplot
                                        sns.boxplot(data=prob_data, palette=['red', 'orange', 'green'], ax=ax)
                                        ax.set_ylabel('Probability')
                                        ax.set_title('Distribution of Sentiment Probabilities')
                                        st.pyplot(fig)

                                    with tab3:
                                        st.subheader("Summary Statistics")

                                        # Create summary stats
                                        summary = {
                                            'Total Reviews': len(results_df),
                                            'Positive Reviews': len(results_df[results_df['sentiment'] == 'Positive']),
                                            'Neutral Reviews': len(results_df[results_df['sentiment'] == 'Neutral']),
                                            'Negative Reviews': len(results_df[results_df['sentiment'] == 'Negative']),
                                            'Average Confidence': results_df['confidence'].mean(),
                                            'High Confidence (>0.8) Reviews': len(
                                                results_df[results_df['confidence'] > 0.8]),
                                            'Low Confidence (<0.5) Reviews': len(
                                                results_df[results_df['confidence'] < 0.5])
                                        }

                                        # Display as two columns
                                        col1, col2 = st.columns(2)

                                        for i, (key, value) in enumerate(summary.items()):
                                            if i < len(summary) / 2:
                                                col1.metric(key, f"{value:.1%}" if 'Confidence' in key else value)
                                            else:
                                                col2.metric(key, f"{value:.1%}" if 'Confidence' in key else value)

                                        # Display sentiment percentages
                                        st.subheader("Sentiment Breakdown")
                                        sentiment_pcts = results_df['sentiment'].value_counts(
                                            normalize=True).sort_index()

                                        # Ensure all sentiments are in the index
                                        for sentiment in ['Negative', 'Neutral', 'Positive']:
                                            if sentiment not in sentiment_pcts.index:
                                                sentiment_pcts[sentiment] = 0
                                        sentiment_pcts = sentiment_pcts.reindex(['Negative', 'Neutral', 'Positive'])

                                        # Create dataframe for display
                                        summary_df = pd.DataFrame({
                                            'Sentiment': sentiment_pcts.index,
                                            'Count': results_df['sentiment'].value_counts().reindex(
                                                sentiment_pcts.index).fillna(0).astype(int),
                                            'Percentage': sentiment_pcts.values * 100
                                        })
                                        summary_df['Percentage'] = summary_df['Percentage'].map('{:.1f}%'.format)

                                        st.table(summary_df)

                                    # Download options
                                    st.subheader("Download Results")

                                    col1, col2 = st.columns(2)

                                    with col1:
                                        # Download full results
                                        csv = results_df.to_csv(index=False)
                                        st.download_button(
                                            label="Download Full Results (CSV)",
                                            data=csv,
                                            file_name="sentiment_analysis_results.csv",
                                            mime="text/csv"
                                        )

                                    with col2:
                                        # Download summary report
                                        summary_md = f"""# Sentiment Analysis Summary Report

## Analysis Overview
- **Total Reviews Analyzed:** {summary['Total Reviews']}
- **Date of Analysis:** {pd.Timestamp.now().strftime('%Y-%m-%d')}

## Sentiment Distribution
- **Positive:** {summary['Positive Reviews']} ({summary['Positive Reviews'] / summary['Total Reviews']:.1%})
- **Neutral:** {summary['Neutral Reviews']} ({summary['Neutral Reviews'] / summary['Total Reviews']:.1%})
- **Negative:** {summary['Negative Reviews']} ({summary['Negative Reviews'] / summary['Total Reviews']:.1%})

## Confidence Analysis
- **Average Confidence Score:** {summary['Average Confidence']:.2f}
- **High Confidence Reviews (>0.8):** {summary['High Confidence (>0.8) Reviews']} ({summary['High Confidence (>0.8) Reviews'] / summary['Total Reviews']:.1%})
- **Low Confidence Reviews (<0.5):** {summary['Low Confidence (<0.5) Reviews']} ({summary['Low Confidence (<0.5) Reviews'] / summary['Total Reviews']:.1%})

This report was generated using a machine learning sentiment analysis model.
                                        """

                                        st.download_button(
                                            label="Download Summary Report (MD)",
                                            data=summary_md,
                                            file_name="sentiment_analysis_summary.md",
                                            mime="text/markdown"
                                        )
                except pd.errors.EmptyDataError:
                    st.error("The uploaded file is empty.")
                except pd.errors.ParserError:
                    st.error("Error parsing the CSV file. Please ensure it's a valid CSV format.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.exception(e)  # This will show the full traceback in development

    elif page == "Train Model":
        st.header("Train Sentiment Analysis Model")

        st.write("Upload a CSV file with 'review_text' and 'rating' columns to train the model.")
        st.write("Ratings should be on a scale of 1-5, where:")
        st.write("- 1-2: Negative sentiment")
        st.write("- 3: Neutral sentiment")
        st.write("- 4-5: Positive sentiment")

        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

        if uploaded_file is not None:
            # Save the uploaded file
            if not os.path.exists('uploads'):
                os.makedirs('uploads')

            file_path = os.path.join('uploads', 'training_data.csv')
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Preview the data
            df = pd.read_csv(file_path)
            st.subheader("Data Preview")
            st.dataframe(df.head())

            if 'review_text' not in df.columns or 'rating' not in df.columns:
                st.error("Dataset must contain 'review_text' and 'rating' columns")
            else:
                # Show dataset statistics
                st.subheader("Dataset Statistics")

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Total reviews: {len(df)}")
                    rating_counts = df['rating'].value_counts().sort_index()
                    st.write("Rating distribution:")
                    st.write(rating_counts)

                with col2:
                    # Show rating distribution chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    rating_counts.plot(kind='bar', ax=ax)
                    ax.set_xlabel('Rating')
                    ax.set_ylabel('Count')
                    ax.set_title('Rating Distribution')
                    st.pyplot(fig)

                if st.button("Train Model"):
                    with st.spinner("Training model... This may take a while."):
                        try:
                            # Train the model
                            result = train_model(file_path)

                            # Display training results
                            st.success(f"Model trained successfully with accuracy: {result['accuracy']:.2%}")

                            st.subheader("Classification Report")
                            st.text(result['report'])

                            # Display confusion matrix
                            st.subheader("Confusion Matrix")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                                        xticklabels=['Negative', 'Neutral', 'Positive'],
                                        yticklabels=['Negative', 'Neutral', 'Positive'])
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            st.pyplot(fig)

                            # Update model loaded status
                            model_loaded = True

                            st.success("Model saved successfully! You can now use it for analysis.")
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")


if __name__ == "__main__":
    main()
