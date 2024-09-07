import pandas as pd



def check_accuracy(df):
    # Assuming column names for actual and predicted answers are 'Actual_Answers' and 'Predicted_Answers'
    actual_answers = df['Correct_answer']
    predicted_answers = df['extracted_keyword']

    # Calculate accuracy by comparing actual and predicted answers
    correct_predictions = (actual_answers == predicted_answers).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions
    
    return accuracy
# Read the CSV file into a DataFrame

csv_path = r'C:\Users\Shree\Desktop\projects\Question Generation\tfidf.csv'

tfidf = pd.read_csv(csv_path)
tfidf_score = check_accuracy(tfidf)
print(f"Accuracy of TFIDF Vectorizer: {tfidf_score * 100:.2f}%")

csv_path = r'C:\Users\Shree\Desktop\projects\Question Generation\bert.csv'

bert = pd.read_csv(csv_path)
bert_score = check_accuracy(bert)
print(f"Accuracy of BERT Attention Mechanism: {bert_score * 100:.2f}%")

csv_path = r'C:\Users\Shree\Desktop\projects\Question Generation\pos.csv'

pos = pd.read_csv(csv_path)
pos_score = check_accuracy(pos)
print(f"Accuracy of POS Tagging: {pos_score * 100:.2f}%")

csv_path = r'C:\Users\Shree\Desktop\projects\Question Generation\rake.csv'

rake = pd.read_csv(csv_path)
rake_score = check_accuracy(rake)
print(f"Accuracy of RAKE Algorithm: {rake_score * 100:.2f}%")

csv_path = r'C:\Users\Shree\Desktop\projects\Question Generation\yake.csv'

yake = pd.read_csv(csv_path)
yake_score = check_accuracy(yake)
print(f"Accuracy of YAKE algorithm: {yake_score * 100:.2f}%")

csv_path = r'C:\Users\Shree\Desktop\projects\Question Generation\kpt.csv'

kpt = pd.read_csv(csv_path)
kpt_score = check_accuracy(kpt)
print(f"Accuracy of keyphrase Transformer: {kpt_score * 100:.2f}%")