import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# 1. Import Libraries (done above)

# 2. Load Data
df = pd.read_csv('../../OneDrive/Masaüstü/trae2/social_media_comments.csv', encoding='windows-1254')
print('Data loaded. Shape:', df.shape)
print(df.head())

# Use actual column names from CSV
comment_col = 'Paylaşım'
sentiment_col = 'Tip'

# 3. Feature Engineering
print('\nValue counts for each column:')
for col in df.columns:
    print(f'\nColumn: {col}')
    print(df[col].value_counts())
    # Value counts bar plot for categorical columns
    if df[col].nunique() < 20 and df[col].dtype == 'object':
        plt.figure(figsize=(8,4))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Value Counts for {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()

print('\nColumns with all unique values:')
unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
print(unique_cols if unique_cols else 'None')

# 4. Explanatory Data Analysis
# 4.1 Sentiment Analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
if comment_col in df.columns:
    df['sentiment_score'] = df[comment_col].astype(str).apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment'] = df['sentiment_score'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))
    print(df['sentiment'].value_counts())
    # Sentiment distribution bar plot
    plt.figure(figsize=(6,4))
    sns.countplot(x='sentiment', data=df, order=['positive','neutral','negative'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()
else:
    print(f'No "{comment_col}" column found for sentiment analysis.')

# 4.2 Common Words
def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(' '.join(text))
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

if comment_col in df.columns:
    plot_wordcloud(df[comment_col].astype(str), 'All Comments - Common Words')
    # 4.3 Positive Common Words
    plot_wordcloud(df[df['sentiment']=='positive'][comment_col].astype(str), 'Positive Comments - Common Words')
    # 4.4 Neutral Common Words
    plot_wordcloud(df[df['sentiment']=='neutral'][comment_col].astype(str), 'Neutral Comments - Common Words')
    # 4.5 Negative Common Words
    plot_wordcloud(df[df['sentiment']=='negative'][comment_col].astype(str), 'Negative Comments - Common Words')
else:
    print(f'No "{comment_col}" column for word cloud analysis.')

# 5. Data Preparation
# 5.1 Split Data
if comment_col in df.columns:
    X = df[comment_col].astype(str)
    if 'sentiment' in df.columns:
        y = df['sentiment']
    else:
        y = df[sentiment_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
else:
    print(f'Required column {comment_col} for modeling not found.')
    X_train_vec = X_test_vec = y_train = y_test = None

# 6. Modeling
results = {}
if X_train_vec is not None:
    # 6.1 Passive Aggressive Classifier
    pac = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    pac.fit(X_train_vec, y_train)
    y_pred_pac = pac.predict(X_test_vec)
    results['PassiveAggressive'] = accuracy_score(y_test, y_pred_pac)
    # 6.2 Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_vec, y_train)
    y_pred_lr = lr.predict(X_test_vec)
    results['LogisticRegression'] = accuracy_score(y_test, y_pred_lr)
    # 6.3 Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_vec, y_train)
    y_pred_rf = rf.predict(X_test_vec)
    results['RandomForest'] = accuracy_score(y_test, y_pred_rf)
    # 6.4 SVM
    svm = LinearSVC(random_state=42)
    svm.fit(X_train_vec, y_train)
    y_pred_svm = svm.predict(X_test_vec)
    results['SVM'] = accuracy_score(y_test, y_pred_svm)
    # 6.5 Multinomial NB
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    y_pred_nb = nb.predict(X_test_vec)
    results['MultinomialNB'] = accuracy_score(y_test, y_pred_nb)
    print('\nModel Accuracies:')
    for model, acc in results.items():
        print(f'{model}: {acc:.4f}')
    # Model accuracy comparison bar plot
    plt.figure(figsize=(8,4))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.show()
    # 7. Best Modeling
    best_model_name = max(results, key=results.get)
    print(f'\nBest Model: {best_model_name}')
    if best_model_name == 'PassiveAggressive':
        best_model = pac
        y_pred_best = y_pred_pac
    elif best_model_name == 'LogisticRegression':
        best_model = lr
        y_pred_best = y_pred_lr
    elif best_model_name == 'RandomForest':
        best_model = rf
        y_pred_best = y_pred_rf
    elif best_model_name == 'SVM':
        best_model = svm
        y_pred_best = y_pred_svm
    else:
        best_model = nb
        y_pred_best = y_pred_nb
    # 7.1 Hyperparameters
    print('\nBest Model Hyperparameters:')
    print(best_model.get_params())
    # 7.2 Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.show()
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred_best))
else:
    print('Modeling skipped due to missing data.')