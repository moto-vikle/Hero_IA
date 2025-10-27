import os
import json
import openai
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

# Importar módulos locales
from modules.Weather import Weather
from modules.llm import LLM
from modules.tts import TTS
from modules.pc_command import PcCommand
from modules.transcriber import Transcriber
from modules.learning_ai import LearningAI  # 🆕 Nuevo módulo

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")

app = Flask(__name__)

# 🆕 Instancia global de IA que aprende
learning_ai = LearningAI()

@app.route("/")
def index():
    return render_template("recorder.html")

@app.route("/audio", methods=["POST"])
def audio():
    audio = request.files.get("audio")
    text = Transcriber().transcribe(audio)

    llm = LLM()
    function_name, args, message = llm.process_functions(text)

    # 🆕 Si no es una función específica, usar IA con aprendizaje
    if function_name is None:
        final_response = learning_ai.process_query(text)
        tts_file = TTS().process(final_response)
        return {
            "result": "ok", 
            "raw_text": text,
            "text": final_response, 
            "file": tts_file,
            "learned": True  # Indicador de que usó aprendizaje
        }

    # Funciones específicas
    if function_name == "get_weather":
        function_response = Weather().get(args["ubicacion"])
        function_response = json.dumps(function_response)
        final_response = llm.process_response(text, message, function_name, function_response)
        
        # 🆕 Aprender de esta interacción
        learning_ai.learn_from_conversation(text, final_response)
        
        tts_file = TTS().process(final_response)
        return {"result": "ok", "raw_text": text, "text": final_response, "file": tts_file}

    elif function_name == "send_email":
        final_response = "Función de envío de correo aún no implementada."
        tts_file = TTS().process(final_response)
        return {"result": "ok", "raw_text": text, "text": final_response, "file": tts_file}

    elif function_name == "open_chrome":
        PcCommand().open_chrome(args["website"])
        final_response = f"Listo, abrí Chrome en {args['website']}"
        
        # 🆕 Aprender de esta interacción
        learning_ai.learn_from_conversation(text, final_response)
        
        tts_file = TTS().process(final_response)
        return {"result": "ok", "raw_text": text, "text": final_response, "file": tts_file}

    final_response = "No entendí tu solicitud."
    tts_file = TTS().process(final_response)
    return {"result": "ok", "raw_text": text, "text": final_response, "file": tts_file}


# 🆕 Endpoints para gestionar el conocimiento
@app.route("/knowledge/stats", methods=["GET"])
def knowledge_stats():
    """Ver estadísticas del aprendizaje"""
    stats = learning_ai.get_statistics()
    return jsonify(stats)

@app.route("/knowledge/search", methods=["POST"])
def knowledge_search():
    """Buscar en la base de conocimiento"""
    data = request.json
    query = data.get("query", "")
    results = learning_ai.semantic_search(query)
    return jsonify(results)

@app.route("/knowledge/reset", methods=["POST"])
def knowledge_reset():
    """Resetear base de conocimiento (usar con cuidado)"""
    learning_ai.knowledge_base = {
        "facts": [],
        "learned_responses": [],
        "user_preferences": {}
    }
    learning_ai.save_knowledge()
    return jsonify({"status": "reset_complete"})

@app.route("/feedback", methods=["POST"])
def feedback():
    """Recibir feedback del usuario para mejorar"""
    data = request.json
    user_input = data.get("user_input")
    ai_response = data.get("ai_response")
    feedback_type = data.get("feedback")  # 'positive' o 'negative'
    
    learning_ai.learn_from_conversation(user_input, ai_response, feedback_type)
    
    return jsonify({"status": "feedback_received"})


if __name__ == "__main__":
    print("🤖 IA con Aprendizaje Autónomo iniciada")
    print(f"📊 Conocimiento inicial: {learning_ai.get_statistics()}")
    app.run(host="0.0.0.0", port=5000, debug=True)