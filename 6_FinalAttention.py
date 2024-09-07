from transformers import BertForTokenClassification, BertTokenizer
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name)

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
csv_path = r'C:\Users\Shree\Desktop\projects\Question Generation\extract_keywords_dataset.csv'

# Text input
text = " Green plants are known as producers because they prepare their own food . The resources which are drawn from nature and used without much modification are known as natural resources . Starch is also called complex carbohydrate . Federalism is the prime feature of our Constitution which refers to the existence of more than one level of government in the country . Deer eats only plant products and so, is called Herbivore ."
# Sample function to generate questions from a passage
def generate_question_from_text(text):
    text = add_space_before_punctuation(text)
    keywords = bert_attention(text)
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
def bert_attention(text):
    # Tokenize input text
    tokens = tokenizer(text, return_tensors='pt')

    # Forward pass through BERT
    outputs = model(**tokens, output_attentions=True)

    # Retrieve attention scores from all layers
    layer_attention = outputs.attentions  # Access attention scores from all layers

    # Find the word with the highest attention score excluding special tokens
    max_attention_word = None
    max_attention_score = 0.0

    # Exclude [CLS] and [SEP] tokens by using the 'attention_mask'
    attention_mask = tokens['attention_mask']
    for layer in range(len(layer_attention)):
        for word_idx, attention_scores in enumerate(layer_attention[layer][0]):
            if attention_mask[0][word_idx] == 1:  # Exclude padding tokens
                max_attention_score = 0.0
                max_attention_word = ""
                avg_attention = torch.mean(attention_scores)
                if avg_attention > max_attention_score:
                    max_attention_score = avg_attention
                    max_attention_word = tokenizer.decode(tokens['input_ids'][0][word_idx].item())
                    print(max_attention_word)
                # Display the original word and its attention score
                original_text_words = text.split()
                if word_idx < len(original_text_words):  # Check if word_idx is within the valid range
                    print(f"The word with the highest attention score is '{original_text_words[word_idx]}' with an attention score of {max_attention_score:.4f}")
                    keyword = original_text_words[word_idx]

# # Tokenize input text
# tokens = tokenizer(text, return_tensors='pt')

# # Forward pass through BERT
# outputs = model(**tokens, output_attentions=True)

# # Retrieve attention scores from all layers
# layer_attention = outputs.attentions  # Access attention scores from all layers

# # Find the word with the highest attention score excluding special tokens
# max_attention_word = None
# max_attention_score = 0.0

# # Exclude [CLS] and [SEP] tokens by using the 'attention_mask'
# attention_mask = tokens['attention_mask']
# for layer in range(len(layer_attention)):
#     for word_idx, attention_scores in enumerate(layer_attention[layer][0]):
#         if attention_mask[0][word_idx] == 1:  # Exclude padding tokens
#             avg_attention = torch.mean(attention_scores)
#             if avg_attention > max_attention_score:
#                 max_attention_score = avg_attention
#                 max_attention_word = tokenizer.decode(tokens['input_ids'][0][word_idx].item())

# # Display the original word and its attention score
# original_text_words = text.split()
# print(f"The word with the highest attention score is '{original_text_words[word_idx]}' with an attention score of {max_attention_score:.4f}")
# keyword = original_text_words[word_idx]
# print(keyword)
# keywords = []
# keywords.append(keyword)
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