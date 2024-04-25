import matplotlib.pyplot as plt
import pandas as pd
import pyttsx3
import seaborn as sns
import speech_recognition
from nltk import download
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Download NLTK data
download('stopwords')

# Load data from text file
with open(r'C:\Users\sunid\OneDrive\Phishing call tone analyzer\dataset.txt', 'r') as file:
    lines = file.readlines()

data = []
for line in lines:
    words = line.strip().split(maxsplit=1)
    if len(words) >= 2:
        label = words[0].strip()
        content = words[1].strip()
        data.append((label, content))

# Convert data to DataFrame
df = pd.DataFrame(data, columns=['label', 'content'])

# Preprocess data
def preprocess_text(text):
    # Add your preprocessing steps here
    return text

df['content'] = df['content'].apply(preprocess_text)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.3, random_state=42)

# Vectorize text data
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Train SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)

# Train Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Evaluate classifiers on test set
X_test_counts = vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

svm_predictions = svm_classifier.predict(X_test_tfidf)
nb_predictions = nb_classifier.predict(X_test_tfidf)

# Print confusion matrices
svm_conf_matrix = confusion_matrix(y_test, svm_predictions)
nb_conf_matrix = confusion_matrix(y_test, nb_predictions)

print("SVM Confusion Matrix:")
print(svm_conf_matrix)
print("\nNaive Bayes Confusion Matrix:")
print(nb_conf_matrix)

# Plot confusion matrices
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.heatmap(svm_conf_matrix, annot=True, fmt="d")
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 2, 2)
sns.heatmap(nb_conf_matrix, annot=True, fmt="d")
plt.title('Naive Bayes Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()


# Initialize speech recognition and text-to-speech
def spk(command):
    voice=pyttsx3.init()
    voice.say(command)
    voice.runAndWait()

sr=speech_recognition.Recognizer()

# Perform speech recognition
with speech_recognition.Microphone() as source2:
    sr.adjust_for_ambient_noise(source2,duration=2)
    print("Speak now..")
    audio2=sr.listen(source2)

    textt=sr.recognize_google(audio2)
    textt=textt.lower()

    print(textt)

# Perform scam call detection on recognized speech
scam_prediction = svm_classifier.predict(vectorizer.transform([preprocess_text(textt)]))[0]

# Output the scam prediction
print(f"Scam Call Detection Prediction: {scam_prediction}")

# If the detected audio is a scam call
if 'fraud' in scam_prediction.lower():
    # Ask the user if they trust the call
    trust_input = input("Do you trust this call? (yes/no): ").lower()

    # If the user trusts the call
    if trust_input == 'yes':
        print("Thanks for your confirmation.")
    # If the user does not trust the call
    else:
        # Terminate the call and block the number
        print("The call is terminated \nThe number has been blocked.")
else:
    print("The call is not detected as a scam.")

# Wait for user input before exiting
input("Press Enter to exit...")
