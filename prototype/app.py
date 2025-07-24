import pandas as pd
import ast
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

# --- 1. Model Training Function (from the research paper) ---
# This function will train the model once when the script starts.
def train_emotion_model():
    """
    Loads the goemotions data, processes it,
    and trains a multi-label classification model.
    """
    print("Loading data and training model... Please wait.")
    
    # Load and process data
    try:
        df = pd.read_csv('goemotions_processed.csv')
    except FileNotFoundError:
        print("FATAL ERROR: 'goemotions_processed.csv' not found.")
        print("Please make sure the dataset is in the same directory as this script.")
        return None, None

    df['labels_list'] = df['labels'].apply(ast.literal_eval)

    # Use MultiLabelBinarizer to one-hot encode the labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['labels_list'])
    X = df['text']

    # Define and train the model pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)))
    ])
    pipeline.fit(X, y)

    print("✅ Model training complete!")
    
    # Return both the trained pipeline and the class names
    return pipeline, mlb.classes_

# --- 2. Train the model globally ---
# This ensures the model is trained only once when the app launches.
MODEL, CLASSES = train_emotion_model()

# --- 3. Prediction Function for the App ---
def predict_emotions(text_input):
    """
    Takes a text string as input and returns the predicted emotions.
    """
    if not MODEL:
        return "Error: Model is not loaded. Please check the console."
    
    # The model expects a list of documents
    predictions = MODEL.predict([text_input])
    
    # Find the labels where the prediction is 1
    predicted_labels = [label for label, prediction in zip(CLASSES, predictions[0]) if prediction == 1]
    
    if not predicted_labels:
        return "Neutral (or no strong emotion detected)"
    else:
        # Join the labels into a nice, readable string
        return ", ".join(predicted_labels).title()

# --- 4. Create and Launch the Gradio Interface ---
if MODEL:
    # Define the interface components
    iface = gr.Interface(
        fn=predict_emotions,
        inputs=gr.Textbox(lines=5, label="Enter Your Text Here", placeholder="Type a sentence to analyze..."),
        outputs=gr.Textbox(label="Predicted Emotions"),
        title="📝 Multi-Label Emotion Classifier",
        description="This is a real-time demo of the model described in the paper 'Beyond BERT: An Efficient and Interpretable Baseline...'. "
                    "It uses a weighted Logistic Regression model to predict 28 different emotions from text.",
        allow_flagging="never",
        examples=[
            ["I am so grateful for all the help you've given me. Thank you!"],
            ["Oh, fantastic. Another last-minute meeting. I am just thrilled."],
            ["This new policy is a total disaster. I can't believe they went through with it."]
        ]
    )

    # Launch the web application
    print("\n🚀 Launching the application... Open the local URL in your browser.")
    iface.launch()