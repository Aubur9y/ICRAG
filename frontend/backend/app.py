from flask import Flask, jsonify, request
from pipeline.orchestrator import pipeline
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route("/apis/query", methods=['POST'])
def query():
    try:
        data = request.get_json()
        user_query = data.get('user_query')
        if not user_query:
            return jsonify({
                "status": "error",
                "message": "Missing user_query in request"
            }), 400
        result = pipeline(user_query)
        return jsonify({
            "status": "success",
            "data": result
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route()

if __name__ == "__main__":
    app.run(debug=True)
