import torch
from datasets import load_dataset
from utils import *
from transformers import TapasConfig, TapasForQuestionAnswering
from transformers import pipeline
from torch.utils.data import DataLoader
from transformers import TapasTokenizer


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



dataset = load_dataset("Stanford/wikitablequestions",trust_remote_code=True)
print("Dataset loaded")
train_data = dataset['train']
test_data = dataset['test']
validation_data = dataset['validation']

train_data = categorize_relations(dataset['train'])
test_data = categorize_relations(dataset['test'])
validation_data = categorize_relations(dataset['validation'])


train_data["table"] = train_data["table"].apply(to_pandas)
test_data["table"] = test_data["table"].apply(to_pandas)
validation_data["table"] = validation_data["table"].apply(to_pandas)
print('Data categorized')

#######################################

tqa = pipeline(task="table-question-answering", model="google/tapas-base", device=0 if torch.cuda.is_available() else -1)
#tqa = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq", device=0 if torch.cuda.is_available() else -1)
#tqa = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq", device=0 if torch.cuda.is_available() else -1)
#tokenizer = TapasTokenizer.from_pretrained("google/tapas-large-finetuned-wtq")

print('Model loaded')



# k_examples = 2
# for i in range(k_examples):
#     item = test_data.iloc[i]  # Get the row as a Series using integer index
#     question = item['question']
#     table = item['table'] # Pass the 'table' column value to to_pandas
#     answer = tqa(table=table, query=question)['answer']

#     print(table.to_markdown())
#     print(f"Question: {question}")
#     print(f"Predicted Answer: {answer}")
#     print("Truth Answer: {0}".format(", ".join(item['answer'])))
#     print("=============================================================================================\n")


# Initialize counters for EM and F1 score
total_examples = 0
correct_predictions = 0
total_f1_score = 0

#lists will contain total_examples, correct_predictions, total_f1_score based on the type
results = {"both":{"total_ex":0,"correct_pred":0,"total_f1":0},
           "header_only":{"total_ex":0,"correct_pred":0,"total_f1":0},
           "table_only":{"total_ex":0,"correct_pred":0,"total_f1":0},
           "none":{"total_ex":0,"correct_pred":0,"total_f1":0}}



batch_size = 124
k = validation_data.shape[0]
print(k)
limit_passes = 0
for i in range(k):
    if i%500 == 0:
      print(f"{i}/{k}")

    item = validation_data.iloc[i] 
    question = item['question']
    
    #print(f"Relation_type: {item['relation_type']}")
    #print(f"Question: {question}")
    table = item['table']
    predicted_answers = []
    if len(table) > tqa.model.config.max_num_rows:
        #print(f"Table with {len(table)} rows exceeds max_num_rows limit of {tqa.model.config.max_num_rows}")
        limit_passes += 1
        for j in range(0,len(table),tqa.model.config.max_num_rows):
            table = table[j:j+tqa.model.config.max_num_rows]
            if len(table) == 0:
                break
            predicted_answer = tqa(table=table, query=question)['answer']
            predicted_answers.append(predicted_answer)
    else: 
        predicted_answer = tqa(table=table, query=question)['answer']
        predicted_answers.append(predicted_answer)

        #continue
    #print(table.to_markdown())
    #predicted_answer = tqa(table=table, query=question)['answer']
    true_answers = item["answer"]
    #print(f"Predicted Answer: {predicted_answer}")
    #print("Truth Answer: {0}".format(", ".join(true_answers)))
    #print("=============================================================================================\n")

    # Calculate EM
    results[item['relation_type']]['total_ex'] += 1
    for predicted_answer in predicted_answers:
        if predicted_answer in true_answers:
            results[item['relation_type']]['correct_pred'] += 1
            correct_predictions += 1
            break
    total_examples += 1

    # Calculate F1 score
    # Assuming true_answers is a list of strings
    # and predicted_answer is a string
    
    # Tokenize predicted and true answers for F1 calculation
    predicted_tokens = predicted_answer.lower().split()
    true_tokens = [token.lower() for answer in true_answers for token in answer.split()]

    # Calculate true positives, false positives, and false negatives
    true_positives = len(set(predicted_tokens) & set(true_tokens))
    false_positives = len(set(predicted_tokens) - set(true_tokens))
    false_negatives = len(set(true_tokens) - set(predicted_tokens))

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    total_f1_score += f1_score
    results[item['relation_type']]['total_f1'] += f1_score

# Calculate overall EM and F1 score
EM = correct_predictions / total_examples
average_f1_score = total_f1_score / total_examples

with open("results/tapas_base_notfinetuned_validation.txt","w") as f:
    # Print the results
    print(f"Total Examples: {total_examples}")
    print(f"EM: {EM:.4f}")
    print(f"Average F1 Score: {average_f1_score:.4f}")
    print(f"Total limit passes: {limit_passes}")
    print("==========================================================================================\n")
    f.write(f"Total Examples: {total_examples}\n")
    f.write(f"EM: {EM:.4f}\n")
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


# Initialize counters for EM and F1 score
# total_examples = 0
# correct_predictions = 0
# total_f1_score = 0

# #lists will contain total_examples, correct_predictions, total_f1_score based on the type
# results = {"both":{"total_ex":0,"correct_pred":0,"total_f1":0},
#            "header_only":{"total_ex":0,"correct_pred":0,"total_f1":0},
#            "table_only":{"total_ex":0,"correct_pred":0,"total_f1":0},
#            "none":{"total_ex":0,"correct_pred":0,"total_f1":0}}



# batch_size = 124
# k = train_data.shape[0]
# print(k)
# limit_passes = 0
# for i in range(k):
#     if i%500 == 0:
#       print(f"{i}/{k}")

#     item = train_data.iloc[i] 
#     question = item['question']
    
#     #print(f"Relation_type: {item['relation_type']}")
#     #print(f"Question: {question}")
#     table = item['table']
#     predicted_answers = []
#     if len(table) > tqa.model.config.max_num_rows:
#         #print(f"Table with {len(table)} rows exceeds max_num_rows limit of {tqa.model.config.max_num_rows}")
#         limit_passes += 1
#         for j in range(0,len(table),tqa.model.config.max_num_rows):
#             table = table[j:j+tqa.model.config.max_num_rows]
#             if len(table) == 0:
#                 break
#             predicted_answer = tqa(table=table, query=question)['answer']
#             predicted_answers.append(predicted_answer)
#     else: 
#         predicted_answer = tqa(table=table, query=question)['answer']
#         predicted_answers.append(predicted_answer)

#         #continue
#     #print(table.to_markdown())
#     #predicted_answer = tqa(table=table, query=question)['answer']
#     true_answers = item["answer"]
#     #print(f"Predicted Answer: {predicted_answer}")
#     #print("Truth Answer: {0}".format(", ".join(true_answers)))
#     #print("=============================================================================================\n")

#     # Calculate EM
#     results[item['relation_type']]['total_ex'] += 1
#     for predicted_answer in predicted_answers:
#         if predicted_answer in true_answers:
#             results[item['relation_type']]['correct_pred'] += 1
#             correct_predictions += 1
#             break
#     total_examples += 1

#     # Calculate F1 score
#     # Assuming true_answers is a list of strings
#     # and predicted_answer is a string
    
#     # Tokenize predicted and true answers for F1 calculation
#     predicted_tokens = predicted_answer.lower().split()
#     true_tokens = [token.lower() for answer in true_answers for token in answer.split()]

#     # Calculate true positives, false positives, and false negatives
#     true_positives = len(set(predicted_tokens) & set(true_tokens))
#     false_positives = len(set(predicted_tokens) - set(true_tokens))
#     false_negatives = len(set(true_tokens) - set(predicted_tokens))

#     # Calculate precision and recall
#     precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
#     recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

#     # Calculate F1 score
#     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
#     total_f1_score += f1_score
#     results[item['relation_type']]['total_f1'] += f1_score

# # Calculate overall EM and F1 score
# EM = correct_predictions / total_examples
# average_f1_score = total_f1_score / total_examples

# with open("results/tapas_base_notfinetuned_train.txt","w") as f:
#     # Print the results
#     print(f"Total Examples: {total_examples}")
#     print(f"EM: {EM:.4f}")
#     print(f"Average F1 Score: {average_f1_score:.4f}")
#     print(f"Total limit passes: {limit_passes}")
#     print("==========================================================================================\n")
#     f.write(f"Total Examples: {total_examples}\n")
#     f.write(f"EM: {EM:.4f}\n")
#     f.write(f"Average F1 Score: {average_f1_score:.4f}\n")
#     f.write("==========================================================================================\n")

#     for relation_type in results:
#         print(f"Type: {relation_type}")
#         print(f"Total Examples: {results[relation_type]['total_ex']}")
#         results[relation_type]['total_ex'] = max(1,results[relation_type]['total_ex'])
#         print(f"EM: {results[relation_type]['correct_pred']/results[relation_type]['total_ex']:.4f}")
#         print(f"Average F1 Score: {results[relation_type]['total_f1']/results[relation_type]['total_ex']:.4f}")
#         print("=============================================================================================\n")
#         f.write(f"Type: {relation_type}\n")
#         f.write(f"Total Examples: {results[relation_type]['total_ex']}\n")
#         f.write(f"EM: {results[relation_type]['correct_pred']/results[relation_type]['total_ex']:.4f}\n")
#         f.write(f"Average F1 Score: {results[relation_type]['total_f1']/results[relation_type]['total_ex']:.4f}\n")
        f.write("=============================================================================================\n")