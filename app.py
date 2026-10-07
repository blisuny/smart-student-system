from flask import Flask, request, jsonify
from flask_cors import CORS
from response_generator import generate_response


app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    """
    Basic route to check if the backend is running.
    """

    return jsonify({
        "message": "Smart Student System backend is running.",
        "status": "success"
    })


@app.route("/ask", methods=["POST"])
def ask():
    """
    Receives the user's question from the frontend,
    sends it to the response generator,
    and returns a natural Turkish answer.
    """

    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({
            "answer": "Lütfen bir soru gönder."
        }), 400

    question = data["question"].strip()

    if question == "":
        return jsonify({
            "answer": "Lütfen boş olmayan bir soru yaz."
        }), 400

    answer = generate_response(question)

    return jsonify({
        "question": question,
        "answer": answer
    })


if __name__ == "__main__":
    app.run(debug=True)