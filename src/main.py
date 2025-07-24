import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

def train_emotion_model():
    """
    Loads the original goemotions data, processes it,
    and trains a multi-label classification model.
    """
    # --- THIS IS THE LINE TO CHECK ---
    # Make sure this filename matches your ORIGINAL uploaded CSV file.
    input_file = 'goemotions_processed.csv'
    # --- END OF LINE TO CHECK ---

    # 1. Load and Validate Data
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"FATAL ERROR: The required data file '{input_file}' was not found.")
        print("Please make sure this file is in the same directory as the script.")
        return None, None

    # Check if the required columns exist
    required_columns = ['text', 'labels']
    if not all(column in df.columns for column in required_columns):
        print(f"FATAL ERROR: The CSV file '{input_file}' is missing required columns.")
        print(f"It must contain both a 'text' and a 'labels' column for training.")
        print("Please use the original 'goemotions_processed.csv' file, not the 'model_ready' version.")
        return None, None

    # Convert the string representation of lists into actual lists
    df['labels_list'] = df['labels'].apply(ast.literal_eval)

    # Use MultiLabelBinarizer to one-hot encode the labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['labels_list'])
    X = df['text']

    # 2. Define and Train the Model
    print("Training the emotion prediction model...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)))
    ])
    pipeline.fit(X, y)
    print("Model training complete! Ready to predict.")

    return pipeline, mlb.classes_

def predict_emotions(text, model, classes):
    """
    Predicts emotions for a given text string using the trained model.
    """
    predictions = model.predict([text])
    predicted_labels = [label for label, prediction in zip(classes, predictions[0]) if prediction == 1]
    return predicted_labels

# Main script execution
if __name__ == "__main__":
    emotion_model, emotion_classes = train_emotion_model()

    if emotion_model and emotion_classes is not None:
        while True:
            user_text = input("\nEnter a sentence to analyze (or type 'quit' to exit): ")
            if user_text.lower() == 'quit':
                print("Exiting the program. Goodbye!")
                break

            if not user_text.strip():
                print("Please enter some text.")
                continue

            predicted_emotions = predict_emotions(user_text, emotion_model, emotion_classes)

            if predicted_emotions:
                print(f"✨ Predicted Emotions: {', '.join(predicted_emotions)}")
            else:
                print("✨ Predicted Emotions: neutral (or no strong emotion detected)")
            print("-" * 30)