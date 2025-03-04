from flask import Flask, render_template, request
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", predictions=[])

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    if file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Load dataset
        df = pd.read_excel(file_path)
        df.drop(columns=["Semester"], inplace=True)
        df = df.astype(bool)

        # Apply Apriori Algorithm
        frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

        # Format output without redundant pairs
        unique_predictions = set()  # Using a set to store unique pairs
        predictions = []

        for _, row in rules.iterrows():
            question1 = ", ".join(sorted(row["antecedents"]))  # Sort questions alphabetically
            question2 = ", ".join(sorted(row["consequents"]))  # Sort consequents alphabetically
            confidence = round(row["confidence"] * 100, 2)

            # Create a unique key (tuple) to check for duplicates
            pair_key = tuple(sorted([question1, question2]))  

            if pair_key not in unique_predictions:
                unique_predictions.add(pair_key)
                predictions.append([question1, question2, confidence])

        return render_template("index.html", predictions=predictions)

if __name__ == "__main__":
    # Get the PORT from environment variables (Render uses dynamic ports)
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=True)
