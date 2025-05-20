from flask import Flask, render_template, request,  jsonify
from dotenv import load_dotenv
from football_agent import chain

load_dotenv()
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/process", methods=["POST"])
def process():
    player_name = request.form["player"]
    try:
        result = chain.invoke({"input": player_name})
        return jsonify(result.dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
