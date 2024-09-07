import random
import spacy
import thinc
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
csv_path = r'C:\Users\Shree\Desktop\projects\Question Generation\bert.csv'

# Sample text
text = "Transformers are our best current solution to contextualization . The type of attention used in them is called self-attention. This mechanism relates different positions of a single sequence to compute a representation of the same sequence. It is instrumental in machine reading, abstractive summarization, and even image description generation. Since they were used initially for machine translation, the Transformers are based on the encoder-decoder architecture, meaning that they have two major components. The first component is an encoder which takes a sequence as input and transforms it into a state with a fixed shape. The second component is the decoder. It maps the encoded state of a fixed shape to an output sequence."

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample function to generate questions from a passage
def generate_question_from_text(text):
    text = add_space_before_punctuation(text)
    # Process the text with spaCy
    doc = nlp(text)

    # Extract keywords using NER and POS tagging
    keywords = set()  # Use a set to eliminate duplicates

    # Extract keywords from Named Entities
    for entity in doc.ents:
        keywords.add(entity.text)

    # Extract keywords based on POS tags (e.g., nouns and proper nouns)
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            keywords.add(token.text)
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
        



