import pandas as pd

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


def to_pandas(item):
  return pd.DataFrame(item["rows"],columns=item["header"])


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))