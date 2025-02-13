from transformers import TapexTokenizer, BartForConditionalGeneration
import torch
from datasets import load_dataset
import pandas as pd
from utils import *

# Load the dataset
dataset = load_dataset("Stanford/wikitablequestions", trust_remote_code=True)
print("Dataset loaded")

# Categorize the data
train_data = categorize_relations(dataset['train'])
test_data = categorize_relations(dataset['test'])
validation_data = categorize_relations(dataset['validation'])

# Convert tables to pandas DataFrame
train_data["table"] = train_data["table"].apply(to_pandas)
test_data["table"] = test_data["table"].apply(to_pandas)
validation_data["table"] = validation_data["table"].apply(to_pandas)
print("Data categorized")

# Load TAPEX model and tokenizer
tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model loaded")

# Initialize counters for EM and F1 score
total_examples = 0
correct_predictions = 0
total_f1_score = 0

# Dictionary to store results for each relation type
results = {
    "both": {"total_ex": 0, "correct_pred": 0, "total_f1": 0},
    "header_only": {"total_ex": 0, "correct_pred": 0, "total_f1": 0},
    "table_only": {"total_ex": 0, "correct_pred": 0, "total_f1": 0},
    "none": {"total_ex": 0, "correct_pred": 0, "total_f1": 0}
}

# Total number of examples in the test data
k = test_data.shape[0]
print(f"Total examples: {k}")
limit_passes = 0

# Iterate through the test data
for i in range(k):
    if i % 500 == 0:
        print(f"Processing {i}/{k} examples...")

    # Get the current example
    item = test_data.iloc[i]
    question = item['question']
    table = item['table']
    true_answers = item["answer"]
    relation_type = item['relation_type']

    # Tokenize inputs
    encoding = tokenizer(table=table, query=question, return_tensors="pt",truncate=True,max_length=1024).to(device)
    # Generate predictions
    with torch.no_grad():
        outputs = model.generate(**encoding)

    # Decode the predicted answer
    predicted_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Update counters for the current relation type
    results[relation_type]['total_ex'] += 1
    total_examples += 1

    # Check if the predicted answer is correct
    if predicted_answer in true_answers:
        results[relation_type]['correct_pred'] += 1
        correct_predictions += 1

    # Calculate F1 score
    predicted_tokens = set(predicted_answer.lower().split())
    true_tokens = set(token.lower() for answer in true_answers for token in answer.split())

    # Calculate true positives, false positives, and false negatives
    true_positives = len(predicted_tokens & true_tokens)
    false_positives = len(predicted_tokens - true_tokens)
    false_negatives = len(true_tokens - predicted_tokens)

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    total_f1_score += f1
    results[relation_type]['total_f1'] += f1

# Calculate overall EM and F1 score
em_score = correct_predictions / total_examples
average_f1_score = total_f1_score / total_examples

with open("results/tapex_test.txt","w") as f:
    # Print the results
    print(f"Total Examples: {total_examples}")
    print(f"EM: {em_score:.4f}")
    print(f"Average F1 Score: {average_f1_score:.4f}")
    print(f"Total limit passes: {limit_passes}")
    print("==========================================================================================\n")
    f.write(f"Total Examples: {total_examples}\n")
    f.write(f"EM: {em_score:.4f}\n")
    f.write(f"Average F1 Score: {average_f1_score:.4f}\n")
    f.write("==========================================================================================\n")

    for relation_type in results:
        print(f"Type: {relation_type}")
        print(f"Total Examples: {results[relation_type]['total_ex']}")
        results[relation_type]['total_ex'] = max(1,results[relation_type]['total_ex'])
        print(f"EM: {results[relation_type]['correct_pred']/results[relation_type]['total_ex']:.4f}")
        print(f"Average F1 Score: {results[relation_type]['total_f1']/results[relation_type]['total_ex']:.4f}")
        print("=============================================================================================\n")
        f.write(f"Type: {relation_type}\n")
        f.write(f"Total Examples: {results[relation_type]['total_ex']}\n")
        f.write(f"EM: {results[relation_type]['correct_pred']/results[relation_type]['total_ex']:.4f}\n")
        f.write(f"Average F1 Score: {results[relation_type]['total_f1']/results[relation_type]['total_ex']:.4f}\n")
        f.write("=============================================================================================\n")


total_examples = 0
correct_predictions = 0
total_f1_score = 0

# Dictionary to store results for each relation type
results = {
    "both": {"total_ex": 0, "correct_pred": 0, "total_f1": 0},
    "header_only": {"total_ex": 0, "correct_pred": 0, "total_f1": 0},
    "table_only": {"total_ex": 0, "correct_pred": 0, "total_f1": 0},
    "none": {"total_ex": 0, "correct_pred": 0, "total_f1": 0}
}

# Total number of examples in the train data
k = train_data.shape[0]
print(f"Total examples: {k}")
limit_passes = 0

# Iterate through the train data
for i in range(k):
    if i % 500 == 0:
        print(f"Processing {i}/{k} examples...")

    # Get the current example
    item = train_data.iloc[i]
    question = item['question']
    table = item['table']
    true_answers = item["answer"]
    relation_type = item['relation_type']

    # Tokenize inputs
    encoding = tokenizer(table=table, query=question, return_tensors="pt",truncate=True,max_length=1024).to(device)
    # Generate predictions
    with torch.no_grad():
        outputs = model.generate(**encoding)

    # Decode the predicted answer
    predicted_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Update counters for the current relation type
    results[relation_type]['total_ex'] += 1
    total_examples += 1

    # Check if the predicted answer is correct
    if predicted_answer in true_answers:
        results[relation_type]['correct_pred'] += 1
        correct_predictions += 1

    # Calculate F1 score
    predicted_tokens = set(predicted_answer.lower().split())
    true_tokens = set(token.lower() for answer in true_answers for token in answer.split())

    # Calculate true positives, false positives, and false negatives
    true_positives = len(predicted_tokens & true_tokens)
    false_positives = len(predicted_tokens - true_tokens)
    false_negatives = len(true_tokens - predicted_tokens)

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    total_f1_score += f1
    results[relation_type]['total_f1'] += f1

# Calculate overall EM and F1 score
em_score = correct_predictions / total_examples
average_f1_score = total_f1_score / total_examples

with open("results/tapex_train.txt","w") as f:
    # Print the results
    print(f"Total Examples: {total_examples}")
    print(f"EM: {em_score:.4f}")
    print(f"Average F1 Score: {average_f1_score:.4f}")
    print(f"Total limit passes: {limit_passes}")
    print("==========================================================================================\n")
    f.write(f"Total Examples: {total_examples}\n")
    f.write(f"EM: {em_score:.4f}\n")
    f.write(f"Average F1 Score: {average_f1_score:.4f}\n")
    f.write("==========================================================================================\n")

    for relation_type in results:
        print(f"Type: {relation_type}")
        print(f"Total Examples: {results[relation_type]['total_ex']}")
        results[relation_type]['total_ex'] = max(1,results[relation_type]['total_ex'])
        print(f"EM: {results[relation_type]['correct_pred']/results[relation_type]['total_ex']:.4f}")
        print(f"Average F1 Score: {results[relation_type]['total_f1']/results[relation_type]['total_ex']:.4f}")
        print("=============================================================================================\n")
        f.write(f"Type: {relation_type}\n")
        f.write(f"Total Examples: {results[relation_type]['total_ex']}\n")
        f.write(f"EM: {results[relation_type]['correct_pred']/results[relation_type]['total_ex']:.4f}\n")
        f.write(f"Average F1 Score: {results[relation_type]['total_f1']/results[relation_type]['total_ex']:.4f}\n")
        f.write("=============================================================================================\n")

