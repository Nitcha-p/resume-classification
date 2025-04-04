'''Step 1 : Import Libraries'''
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

'''Step 2 : Load and Explore the Dataset'''
# Load the dataset
df = pd.read_excel("Extracted_Resumes.xlsx")

# Display basic info
# print(df.info())

# Rename columns
df.rename(columns={"Extracted Text": "Resume_text"}, inplace=True)

# Remove missing values & duplicates
df.drop_duplicates(inplace=True)

'''Step 3 : Data Preprocessing'''
# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Expand contractions
contractions_dict = {
    "I'm": "I am", "can't": "cannot", "won't": "will not", "it's": "it is",
    "he's": "he is", "she's": "she is", "they're": "they are", "you're": "you are",
    "we're": "we are", "isn't": "is not", "aren't": "are not", "doesn't": "does not",
    "didn't": "did not", "haven't": "have not", "hasn't": "has not",
    "shouldn't": "should not", "couldn't": "could not", "wouldn't": "would not",
    "mustn't": "must not", "let's": "let us"
}
def expand_contractions(text):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in contractions_dict.keys()) + r')\b')
    return pattern.sub(lambda x: contractions_dict[x.group()], text)

# Remove personal info (emails, phone numbers, URLs)
def remove_personal_info(text):
    text = re.sub(r'\S+@\S+', ' ', text) # Remove emails
    text = re.sub(r'\b\d{10,15}\b', ' ', text) # Remove phone numbers
    text = re.sub(r'http\S+|www\S+', ' ', text)  # Remove URLs (LinkedIn, websites)
    return text

# General text cleaning
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', ' ', text)  # Remove punctuation, numbers, special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply basic preprocessing
df['Cleaned_Resume_text'] = df['Resume_text'].apply(expand_contractions)
df['Cleaned_Resume_text'] = df['Cleaned_Resume_text'].apply(remove_personal_info)
df['Cleaned_Resume_text'] = df['Cleaned_Resume_text'].apply(clean_text)

# Initialize stop words and Lemmatizer
stop_words = set(ENGLISH_STOP_WORDS)  
lemmatizer = WordNetLemmatizer()

# Final preprocessing function with stopword removal
def nltk_preprocess(text):
    tokens = word_tokenize(text)  # Tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatization + Stopword Removal
    return " ".join(tokens)

# Apply NLTK preprocessing
df['Cleaned_Resume_text'] = df['Cleaned_Resume_text'].apply(nltk_preprocess)

# Show sample data
# print(df.head())

# Save cleaned data to CSV file
# df.to_csv('Cleaned_Resumes.csv', index=False)

'''Step 4 : Check Data Balance'''
# Count the number of resumes per category
category_counts = df["Category"].value_counts()
# print(category_counts)

# Plot class distribution
# plt.figure(figsize=(12, 6))
# sns.barplot(x=category_counts.index, y=category_counts.values, palette="viridis")
# plt.xticks(rotation=45, ha="right")
# plt.xlabel("Job Category")
# plt.ylabel("Number of Resumes")
# plt.title("Resume Category Distribution")
# plt.show()

'''Step 5 : Convert Text to Features (TF-IDF)'''
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=6000, sublinear_tf=True, min_df=2, max_df=0.9)
X_tfidf = tfidf.fit_transform(df['Cleaned_Resume_text'])

'''Step 6 : Encode Job Categories'''
# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Category'])

# Store label mappings
label_mappings = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
# print("Category Mappings :", label_mappings) 

'''Step 7: Chi-Square Feature Selection'''
feature_selector = SelectKBest(chi2, k=500)
X_selected = feature_selector.fit_transform(X_tfidf, y)


'''Step 8 : Split Data for Training and Testing'''
# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

'''Step 9  : Use Grid Search for Hyperparameter Tuning'''
from sklearn.model_selection import GridSearchCV

# Define hyperparameters grid
param_grid = {
    'C': [0.005, 0.001, 1],  # Regularization strength
    'loss': ['hinge', 'squared_hinge'],  # Type of loss function
    'max_iter': [1000, 3000, 5000]  # Maximum number of iterations
}

# Perform Grid Search
grid_search = GridSearchCV(LinearSVC(class_weight='balanced'), param_grid, cv=5, scoring='recall_macro', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Display best parameters
print("Best Parameters:", grid_search.best_params_)

'''Step 10 : Train Model'''
# Train model with best parameters
best_svm = LinearSVC(**grid_search.best_params_)
best_svm.fit(X_train, y_train)

'''Step 11 : Evaluate Model Performance'''
# Predict on test data
y_pred = best_svm.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Evaluate on training set to check for overfitting
y_train_pred = best_svm.predict(X_train)
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred, target_names=label_encoder.classes_))

'''After'''
# Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = label_encoder.classes_

# Plot Confusion Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix for Resume Classification")
plt.show()


