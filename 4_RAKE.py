# extract keywords

from rake_nltk import Rake

# Create a Rake object
r = Rake()
# text cleaning

# 1. lowercasing
def lowercase(text):
    ans = text.lower()
    return ans

# 2. Removing Punctuations
import string
def remove_punctuations(text):
    PUNCT_TO_REMOVE = string.punctuation
    ans = text.translate(str.maketrans('','', PUNCT_TO_REMOVE))
    return ans

# 3. Removing Extra Space
def remove_extra_space(text):
    ans = " ".join(text.split())
    return ans

# 4. Removing Contractions
import contractions
def remove_contractions(text):
    contractions.fix(text)
    return text

# 5. Remove HTML tag
from bs4 import BeautifulSoup

# Assuming 'html_text' contains HTML data
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    return text

#6. Remove Special Characters and Punctuation
import re
def remove_special_char_n_punct(text):
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text

#7. Stopword Removal
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(tokens)
    return text

def preprocess(text):
    text = lowercase(text)
    # text = remove_punctuations(text)
    text = remove_extra_space(text)
    text = remove_contractions(text)
    return text
import string
def add_space_before_punctuation(input_string):
    punctuations = set(string.punctuation)
    modified_string = ''
    
    for char in input_string:
        if char in punctuations:
            modified_string += ' ' + char
        else:
            modified_string += char
    
    return modified_string
import pandas as pd
import csv
dataset = pd.read_csv('extract_keywords_dataset.csv', encoding='ISO-8859-1')
csv_path = r'C:\Users\Shree\Desktop\projects\Question Generation\rake.csv'

# Sample function to generate questions from a passage
def generate_question_from_text(text):
    text = add_space_before_punctuation(text)
    text = preprocess(text)
    # Extract keywords using RAKE
    r.extract_keywords_from_text(text)

    # Get the ranked keywords
    keywords = r.get_ranked_phrases()
    result = ""
    word = ""
    ans = ""
    omitted_keywords = []
    keyword_replaced = False  
    keyword_present = True
    for letter in text:
        if letter == " ":
            if word.lower() in keywords:
                if not keyword_replaced:
                    result += " " + "___________" + " "
                    ans = word
                    keyword_replaced = True
                    keyword_present = True
                else:
                    result += " " + word
            else:
                result += " " + word
                if word.lower() in omitted_keywords:
                    omitted_keywords.remove(word.lower())
                omitted_keywords.append(word.lower())
            word = ""
        elif letter == ".":
            if (keyword_present == True):
                result += letter
                # print("Q: " + result)
                # result = ""   
                keyword_present = False
                # print("Ans: " + ans)
                keyword_replaced = False  # Reset the flag for the next line
            else:
                result += ""
                keyword_replaced = False
        else:
            word += letter
    # print(result)         
    return result.strip(), ans

with open(csv_path, 'r', newline='') as file:
    csv_reader = csv.DictReader(file)
    
    rows = list(csv_reader)  # Read all rows into a list

    with open(csv_path, 'a', newline='') as output_file:  # Open the output file for appending
        fieldnames = csv_reader.fieldnames + ['Generated_Output', 'extracted_keyword']  # Include a new column header
        
        csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        csv_writer.writeheader()  # Write the header (if the file is empty)

        for row in rows:
            text = row['Passage']
            
            # Text processing step: Add space before punctuation marks
            text = add_space_before_punctuation(text)
            
            # Generate a question from the processed text
            generated_question, extracted_keyword = generate_question_from_text(text)
            
            # Print the generated question (for demonstration)
            # print(generated_question)
            
            # Update the row with the generated question in the 'Generated_Output' column
            row['Generated_Output'] = generated_question
            row['extracted_keyword'] =  extracted_keyword
            # Write the updated row to the CSV file
            csv_writer.writerow(row)
        




