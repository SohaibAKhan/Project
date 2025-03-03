import logging
import requests
from flask import Flask, request, jsonify

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Hugging Face API Key
HUGGINGFACE_API_KEY = "Your API Key"

# AI-based Code Review using Hugging Face API
def ai_code_review(code):
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "inputs": f"""
        Review this Python code with a focus on:
        - Security risks (e.g., hardcoded credentials, missing validation, unsafe imports).
        - Best coding practices (e.g., modularization, efficiency, clarity).
        - If the code is good, acknowledge it rather than giving unnecessary suggestions.

        Provide the response in JSON format with keys 'issues' and 'suggestions'.

        Code to review:
        {code}
        """
    }

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/bigcode/starcoder",
            json=data,
            headers=headers,
            timeout=10,  # Prevents hanging API calls
        )
        response.raise_for_status()
        ai_response = response.json()

        logging.info(f"Raw AI Response: {ai_response}")

        if not ai_response or not isinstance(ai_response, list):
            return {"error": "Invalid response from AI"}

        review_text = ai_response[0].get("generated_text", "No review text received from AI.")

        # Extracting issues and suggestions
        formatted_review = {"issues": [], "suggestions": []}
        seen_issues = set()
        seen_suggestions = set()

        for line in set(review_text.split("\n")):  # Remove duplicate lines
            clean_line = line.strip("-* ").strip()
            if not clean_line or "Code to review:" in clean_line:
                continue

            if any(keyword in clean_line.lower() for keyword in ["risk", "vulnerability", "error", "issue", "hardcoded"]):
                if clean_line not in seen_issues:
                    formatted_review["issues"].append(clean_line)
                    seen_issues.add(clean_line)

            elif any(keyword in clean_line.lower() for keyword in ["suggest", "consider", "improve", "optimize"]):
                if clean_line not in seen_suggestions:
                    formatted_review["suggestions"].append(clean_line)
                    seen_suggestions.add(clean_line)

        return formatted_review

    except requests.exceptions.Timeout:
        logging.error("Hugging Face API request timed out.")
        return {"error": "AI service timeout, please try again later."}

    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Hugging Face API: {str(e)}")
        return {"error": "Failed to analyze code using AI"}

# File Upload API for Code Review
@app.route("/upload", methods=["POST"])
def upload_code():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not file.filename.endswith(".py"):
            return jsonify({"error": "Invalid file type. Only Python (.py) files are accepted."}), 415

        code = file.read().decode("utf-8")

        # AI-Based Review
        ai_feedback = ai_code_review(code)

        return jsonify({"status": "Code review completed", "ai_feedback": ai_feedback})

    except Exception as e:
        logging.error(f"[{request.remote_addr}] Error processing request: {str(e)}")
        return jsonify({"error": "An internal server error occurred"}), 500

# Run Flask API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)