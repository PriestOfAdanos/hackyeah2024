from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

AI_ASSISTANT_URL = "http://fin-tax-ai:5000/api/assist"
ELASTICSEARCH_URL = "http://fin-tax-elasticsearch:9200"

@app.route('/api/orchestrate', methods=['POST'])
def orchestrate():
    user_input = request.json.get("user_input")
    
    # Orchestrate the call to the AI assistant
    ai_response = requests.post(AI_ASSISTANT_URL, json={"user_input": user_input})
    ai_data = ai_response.json()
    
    # Placeholder for further orchestration, e.g., interaction with Elasticsearch
    es_response = f"Retrieved information related to '{user_input}' from Elasticsearch"

    return jsonify({
        "ai_response": ai_data.get("response"),
        "retrieved_data": es_response
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
