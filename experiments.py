from transformers import TapasForQuestionAnswering, T5ForConditionalGeneration, AutoTokenizer
from transformers import pipeline
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
model = None

# Check correlation of question with headers
def is_question_related_to_header(question, headers):
    """Check if a question relates to table headers."""
    for header in headers:
        if header.lower() in question.lower():
            return True
    return False

# Check correlation of question with row values
def is_question_related_to_row(question, rows):
    """Check if a question relates to table row values."""
    for row in rows:
        for value in row:
            if str(value).lower() in question.lower():
                return True
    return False

# Determine relations
def categorize_relations(data):
    """Categorize each question based on its relation to headers and rows."""
    categorized = []
    for example in data:
        question = example['question']
        headers = example['table']['header']
        rows = example['table']['rows']
        is_header_related = is_question_related_to_header(question, headers)
        is_row_related = is_question_related_to_row(question, rows)
        
        # Determine category
        if is_header_related and is_row_related:
            relation_type = "both"
        elif is_header_related:
            relation_type = "header_only"
        elif is_row_related:
            relation_type = "table_only"
        else:
            relation_type = "none"

        categorized.append({
            "question": question,
            "answer": example['answers'],
            "relation_type": relation_type,
            "table": example['table']
        })

    return pd.DataFrame(categorized)

# Model evaluation function
def evaluate_model(model, dataset):
    """Evaluate a model's performance on a dataset."""
    predictions = []
    for _, row in dataset.iterrows():
        question = row['question']
        table = row['table']

        if isinstance(model, TapasForQuestionAnswering) :
            inputs = tokenizer(
                    table=table,
                    queries=question,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )            
            outputs = model(**inputs)
            predicted_answer = tokenizer.decode(outputs.logits.argmax(dim=-1).squeeze().tolist())
        
        elif isinstance(model, T5ForConditionalGeneration):
            input_text = f"Question: {question} Table: {str(table)}"
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            result = model.generate(**inputs)
            predicted_answer = tokenizer.decode(result[0], skip_special_tokens=True)
        else:
            predicted_answer = model.predict(question)[0]
        predictions.append(predicted_answer)

    match = accuracy_score(dataset["answer"].tolist(), predictions)
    f1 = f1_score(dataset["answer"].tolist(), predictions, average="macro")  # Macro for multi-class
    return match, f1

# Run experiments
def run_experiments(models, categorized_data):
    """Run experiments for multiple models across relation categories."""
    results = {}
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        results[model_name] = {}
        for category, subset in categorized_data.items():
            print(f" - Evaluating category: {category}")
            em, f1 = evaluate_model(model, subset)
            results[model_name][category] = {"EM": em, "F1": f1}
    return results


def load_tapas_model():
        """Loads and initializes the TAPAS model."""
        model_name = "google/tapas-base-finetuned-wtq"  # Example model name
        model = TapasForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer  # Return both model and tokenizer

def load_t5_model():
    """Loads and initializes the T5 model."""
    model_name = "t5-base"  # Example model name
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer  # Return both model and tokenizer

def load_baseline_model():
    """Loads and initializes your baseline model."""
    # Your baseline model loading logic here
    # ...
    return baseline_model

def preprocess_for_tapas(question, table):
    """Preprocess question and table for TAPAS."""
    # Convert table to a format TAPAS can understand
    table_data = {
        "header": table["header"],
        "rows": table["rows"]
    }
    inputs = tokenizer(
        table=table_data,
        queries=question,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    return inputs

# Main
if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset('Stanford/wikitablequestions', "random-split-1",trust_remote_code=True)
    train_data = dataset['train']
    test_data = dataset['test']

    # Categorize training and testing data
    categorized_train = categorize_relations(train_data)
    categorized_test = categorize_relations(test_data)

    # Group data by relation type
    categorized_data_train = {
        "header_only": categorized_train[categorized_train['relation_type'] == "header_only"],
        "table_only": categorized_train[categorized_train['relation_type'] == "table_only"],
        "both": categorized_train[categorized_train['relation_type'] == "both"],
        "none": categorized_train[categorized_train['relation_type'] == "none"]
    }

    categorized_data_test = {
        "header_only": categorized_test[categorized_test['relation_type'] == "header_only"],
        "table_only": categorized_test[categorized_test['relation_type'] == "table_only"],
        "both": categorized_test[categorized_test['relation_type'] == "both"],
        "none": categorized_test[categorized_test['relation_type'] == "none"]
    }

    

    # Define models (Replace with your implementations)
    models = {
        "TAPAS": load_tapas_model(),  
        "T5": load_t5_model(),        
        #"Baseline": load_baseline_model(), 
    }

    # Run experiments
    train_results = run_experiments(models, categorized_data_train)
    test_results = run_experiments(models, categorized_data_test)

    # Display results


    for model_name in models.keys():
        with open(f"results_{model_name}.txt", "w") as f:
            f.write(f"\nResults for {model_name} - Training Set:\n")
            print(f"\nResults for {model_name} - Training Set:")
            for category, metrics in train_results[model_name].items():
                f.write(f"  {category}: EM={metrics['EM']:.2f}, F1={metrics['F1']:.2f}\n")
                print(f"  {category}: EM={metrics['EM']:.2f}, F1={metrics['F1']:.2f}")

            f.write(f"\nResults for {model_name} - Test Set:\n")
            print(f"\nResults for {model_name} - Test Set:")
            for category, metrics in test_results[model_name].items():
                f.write(f"  {category}: EM={metrics['EM']:.2f}, F1={metrics['F1']:.2f}\n")
                print(f"  {category}: EM={metrics['EM']:.2f}, F1={metrics['F1']:.2f}")

