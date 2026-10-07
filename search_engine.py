import re
from data_loader import load_all_data
from intent_detector import detect_intent


def clean_text(text):
    """
    Converts text to lowercase and removes unnecessary characters.
    This helps the search system compare the question with dataset content.
    """

    text = str(text).lower()
    text = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def flatten_value(value):
    """
    Converts different data types into searchable text.
    The dataset contains strings, lists and dictionaries.
    This function turns all of them into one text block.
    """

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return " ".join(flatten_value(item) for item in value)

    if isinstance(value, dict):
        return " ".join(flatten_value(item) for item in value.values())

    return str(value)


def item_to_searchable_text(item):
    """
    Converts one dataset item into a single searchable text.
    """

    text_parts = []

    for value in item.values():
        text_parts.append(flatten_value(value))

    return clean_text(" ".join(text_parts))


def calculate_score(question, item):
    """
    Gives a relevance score to a dataset item.
    Higher score means the item is more related to the question.
    """

    question_clean = clean_text(question)
    searchable_text = item_to_searchable_text(item)

    question_words = question_clean.split()
    score = 0

    for word in question_words:
        if len(word) <= 2:
            continue

        if word in searchable_text:
            score += 2

    # Extra score if the item has a keywords field
    keywords = item.get("keywords", [])

    for keyword in keywords:
        keyword_clean = clean_text(keyword)

        if keyword_clean and keyword_clean in question_clean:
            score += 5

    # Extra score if title/name directly matches the question
    title = clean_text(item.get("title", ""))
    name = clean_text(item.get("name", ""))

    if title and title in question_clean:
        score += 8

    if name and name in question_clean:
        score += 8

    return score


def search_dataset(question, intent, all_data):
    """
    Searches the selected dataset according to the detected intent.
    """

    if intent not in all_data:
        return None

    dataset = all_data[intent]

    best_item = None
    best_score = 0

    for item in dataset:
        score = calculate_score(question, item)

        if score > best_score:
            best_score = score
            best_item = item

    if best_score == 0:
        return None

    return best_item


if __name__ == "__main__":
    all_data = load_all_data()

    test_questions = [
        "Rowanda hoca hangi bölümde?",
        "Erasmus hibesi ne kadar?",
        "Hangi kulüpler var?",
        "Mayıs ayında etkinlik var mı?",
        "Araştırma merkezleri neler?"
    ]

    for question in test_questions:
        intent = detect_intent(question)
        result = search_dataset(question, intent, all_data)

        print("Question:", question)
        print("Intent:", intent)

        if result:
            print("Best Match:", result.get("title") or result.get("name"))
        else:
            print("Best Match: Not found")

        print("-----------------------------")