import json
import os


# Finds the folder where this file is located.
# In this case, it gets the path of the backend folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The folder where the dataset files are stored.
# Since your folder name is "Data", we write "Data" here.
DATA_DIR = os.path.join(BASE_DIR, "Data")


def load_json_file(filename):
    """
    Opens the given JSON file, reads it, and converts it into a Python list.
    For example, it reads professors.json and returns the professor records.
    """

    file_path = os.path.join(DATA_DIR, filename)

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data

    except FileNotFoundError:
        print(f"ERROR: {filename} file was not found.")
        return []

    except json.JSONDecodeError as e:
        print(f"ERROR: {filename} has an invalid JSON format.")
        print(f"Line: {e.lineno}")
        print(f"Column: {e.colno}")
        print(f"Problem: {e.msg}")
        return []


def load_all_data():
    """
    Loads all datasets and stores them in one dictionary.
    This allows the chatbot to access all categories.
    """

    all_data = {
        "professors": load_json_file("professors.json"),
        "erasmus": load_json_file("erasmus.json"),
        "clubs": load_json_file("clubs.json"),
        "events": load_json_file("events.json"),
        "research": load_json_file("research.json")
    }

    return all_data


# This part is only for testing.
# When this file is run directly, it loads the datasets
# and prints how many records each dataset contains.
if __name__ == "__main__":
    data = load_all_data()

    print("Datasets loaded successfully.")
    print("-----------------------------")

    for category, items in data.items():
        print(f"{category}: {len(items)} records")