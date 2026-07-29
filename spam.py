import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download nltk data
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Load dataset 
data = pd.read_csv("D:\Desktop\spam\email.csv", encoding='latin-1')

print("Columns:", data.columns)
print("Missing values:\n", data.isnull().sum())


# Select & rename columns
data = data[['Category', 'Message']]
data.columns = ['label', 'text']


# Clean labels
data['label'] = data['label'].astype(str)
data['label'] = data['label'].str.strip()
data['label'] = data['label'].str.lower()

# Convert labels to numeric
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Remove invalid rows
data = data.dropna(subset=['label'])

# Convert to int
data['label'] = data['label'].astype(int)

print("Label values:", data['label'].unique())


# Text Preprocessing function

def clean_text(text):
    text = text.lower()
    
    # remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    
    # tokenize
    words = text.split()
    
    # remove stopwords
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

# Apply preprocessing
data['text'] = data['text'].apply(clean_text)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)


# TF-IDF 
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)


# Evaluate model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Email input
while True:
    print("\nEnter full email (type END to finish, EXIT to quit):")

    lines = []

    while True:
        line = input()

        if line.strip().upper() == "END":
            break
        if line.strip().upper() == "EXIT":
            print("Exiting...")
            exit()

        lines.append(line)

    # Combine all lines
    msg = " ".join(lines)

    # Preprocess input
    msg_clean = clean_text(msg)
    msg_vec = vectorizer.transform([msg_clean])

    result = model.predict(msg_vec)[0]

    if result == 1:
        print("\nSpam ❌")
    else:
        print("\nNot Spam ✅")

