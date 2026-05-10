# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, Response, jsonify, stream_with_context
import httpx
import json
import os

app = Flask(__name__)

API_KEY = os.environ.get("OLLAMA_API_KEY", "9c4159e43d624f12985c32e10fa11f6c.6ZpOFJvp3JDIQSKSvh4Oe2lA")
API_URL = "https://ollama.com/v1/chat/completions"
MODELS_URL = "https://ollama.com/v1/models"

chat_histories = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/models")
def models():
    headers = {"Authorization": f"Bearer {API_KEY}"}
    try:
        r = httpx.get(MODELS_URL, headers=headers, timeout=15, follow_redirects=True)
        data = r.json()
        model_list = [m["id"] for m in data.get("data", [])]
        return jsonify({"models": model_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    model = data.get("model", "gpt-oss:20b")
    session_id = data.get("session_id", "default")

    if session_id not in chat_histories:
        chat_histories[session_id] = [
            {"role": "system", "content": "You are a helpful and friendly AI assistant. Answer clearly and helpfully."}
        ]

    chat_histories[session_id].append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": chat_histories[session_id],
        "stream": True,
        "temperature": 0.7
    }

    def generate():
        full_reply = ""
        try:
            with httpx.stream("POST", API_URL, json=payload, headers=headers,
                              timeout=60, follow_redirects=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk.strip() == "[DONE]":
                            break
                        try:
                            d = json.loads(chunk)
                            delta = d["choices"][0]["delta"].get("content", "")
                            if delta:
                                full_reply += delta
                                yield f"data: {json.dumps({'content': delta})}\n\n"
                        except Exception:
                            pass
            chat_histories[session_id].append({"role": "assistant", "content": full_reply})
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route("/clear", methods=["POST"])
def clear():
    session_id = request.json.get("session_id", "default")
    chat_histories[session_id] = [
        {"role": "system", "content": "You are a helpful and friendly AI assistant. Answer clearly and helpfully."}
    ]
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
