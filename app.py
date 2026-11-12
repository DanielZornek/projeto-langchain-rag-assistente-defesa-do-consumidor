
import re
import torch
from flask import Flask, jsonify, render_template_string, request
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# =====================================================
# 1. CONFIGURAÇÃO DOS MODELOS
# =====================================================

print("Carregando modelo local...")

MODEL_ID = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
model.eval()

app = Flask(__name__)

# =====================================================
# 2. CARREGAMENTO DO CHROMADB
# =====================================================

print("Carregando banco vetorial...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db_cdc",
    embedding_function=embeddings
)

print("Banco vetorial carregado!")

# =====================================================
# 3. FUNÇÕES AUXILIARES
# =====================================================

def identificar_artigos(texto):
    """
    Identifica números de artigos presentes no trecho.
    Não cria referências: apenas extrai o que está escrito.
    """
    encontrados = re.findall(
        r'\bArt(?:igo)?\.?\s*(\d+[º°]?)',
        texto,
        flags=re.IGNORECASE
    )

    return list(dict.fromkeys(encontrados))


def preparar_contexto(documentos):
    """
    Organiza os trechos recuperados e identifica
    os artigos encontrados em cada um.
    """
    partes = []
    fontes = []

    for i, doc in enumerate(documentos, start=1):
        conteudo = doc.page_content.strip()

        if not conteudo:
            continue

        artigos = identificar_artigos(conteudo)

        identificacao = (
            ", ".join(["Art. " + a for a in artigos])
            if artigos else "Artigo não identificado"
        )

        partes.append(
            f"FONTE {i} ({identificacao}):\n{conteudo}"
        )

        fontes.append({
            "numero": i,
            "artigos": identificacao,
            "conteudo": conteudo[:1200]
        })

    return "\n\n".join(partes), fontes


def gerar_resposta(pergunta, contexto):
    """
    Gera uma resposta usando exclusivamente
    os trechos recuperados do CDC.
    """

    prompt = f"""
Você é um assistente que responde dúvidas sobre o
Código de Defesa do Consumidor brasileiro.

Use somente as informações do CONTEXTO abaixo.

REGRAS:
- Responda em português claro e direto.
- Não invente artigos, leis ou números.
- Não use informações que não estejam no contexto.
- Não diga que um artigo trata de determinado assunto
  se isso não estiver escrito no contexto.
- Se o contexto não permitir responder, diga:
  "Não encontrei informações suficientes nos trechos
  recuperados para responder com segurança."
- Não responda com informações de conhecimento geral.

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    with torch.inference_mode():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=120,
            num_beams=2,
            do_sample=False,
            no_repeat_ngram_size=3
        )

    resposta = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    if not resposta:
        resposta = (
            "Não foi possível gerar uma resposta com "
            "os trechos recuperados."
        )

    return resposta


# =====================================================
# 4. INTERFACE HTML
# =====================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Assistente Virtual do Consumidor</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            background: #f4f4f9;
            color: #333;
        }

        h2 {
            text-align: center;
            color: #222;
        }

        textarea {
            width: 100%;
            height: 100px;
            padding: 12px;
            font-size: 15px;
            border: 1px solid #ccc;
            border-radius: 6px;
            box-sizing: border-box;
            resize: vertical;
        }

        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 13px;
            font-size: 16px;
            border-radius: 6px;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
        }

        button:hover {
            background: #0056b3;
        }

        button:disabled {
            background: #888;
            cursor: wait;
        }

        .caixa {
            margin-top: 20px;
            padding: 18px;
            background: white;
            border-radius: 7px;
            border: 1px solid #ddd;
            white-space: pre-wrap;
            line-height: 1.6;
        }

        #resposta, #fontes {
            display: none;
        }

        .fonte {
            background: #fff;
            border-left: 4px solid #007bff;
            padding: 12px;
            margin-top: 12px;
            font-size: 14px;
            white-space: pre-wrap;
        }

        .loading {
            display: none;
            text-align: center;
            margin-top: 15px;
            color: #555;
        }

        .aviso {
            font-size: 12px;
            color: #666;
            margin-top: 20px;
            text-align: center;
        }

        h3 {
            margin-bottom: 8px;
        }
    </style>
</head>

<body>

    <h2>⚖️ Assistente Virtual do Consumidor</h2>

    <p>
        Faça uma pergunta sobre o Código de Defesa do Consumidor:
    </p>

    <textarea
        id="pergunta"
        placeholder="Ex: Posso desistir de uma compra feita pela internet?"
    ></textarea>

    <button id="botao" onclick="enviarPergunta()">
        Consultar CDC
    </button>

    <div id="loading" class="loading">
        Consultando a base de conhecimento...
    </div>

    <div id="resposta" class="caixa"></div>

    <div id="fontes" class="caixa">
        <h3>Trechos recuperados do CDC</h3>
        <div id="listaFontes"></div>
    </div>

    <p class="aviso">
        Sistema acadêmico de consulta baseado em RAG.
        As respostas dependem dos documentos disponíveis na base.
    </p>

<script>

async function enviarPergunta() {

    const campo = document.getElementById("pergunta");
    const botao = document.getElementById("botao");
    const pergunta = campo.value.trim();

    if (!pergunta) {
        alert("Digite uma pergunta!");
        return;
    }

    botao.disabled = true;
    botao.innerText = "Consultando...";
    document.getElementById("loading").style.display = "block";
    document.getElementById("resposta").style.display = "none";
    document.getElementById("fontes").style.display = "none";

    try {

        const response = await fetch("/perguntar", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                pergunta: pergunta
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.resposta || "Erro no servidor.");
        }

        document.getElementById("resposta").innerText =
            "RESPOSTA:\\n\\n" + data.resposta;

        document.getElementById("resposta").style.display = "block";

        const lista = document.getElementById("listaFontes");
        lista.innerHTML = "";

        if (data.fontes && data.fontes.length > 0) {

            data.fontes.forEach(function(fonte) {

                const div = document.createElement("div");
                div.className = "fonte";

                const titulo = document.createElement("strong");
                titulo.textContent =
                    "Trecho " + fonte.numero + " — " + fonte.artigos;

                const conteudo = document.createElement("p");
                conteudo.textContent = fonte.conteudo;

                div.appendChild(titulo);
                div.appendChild(conteudo);
                lista.appendChild(div);
            });

            document.getElementById("fontes").style.display = "block";
        }

    } catch (error) {

        document.getElementById("resposta").innerText =
            "Erro: " + error.message;

        document.getElementById("resposta").style.display = "block";

    } finally {

        document.getElementById("loading").style.display = "none";
        botao.disabled = false;
        botao.innerText = "Consultar CDC";
    }
}

</script>

</body>
</html>
"""


# =====================================================
# 5. ROTAS FLASK
# =====================================================

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route("/perguntar", methods=["POST"])
def perguntar():

    try:
        dados = request.get_json(silent=True) or {}
        pergunta = dados.get("pergunta", "").strip()

        if not pergunta:
            return jsonify({
                "resposta": "Digite uma pergunta válida.",
                "fontes": []
            }), 400

        # Recupera os 5 trechos mais próximos da pergunta
        documentos_com_distancia = (
            vectorstore.similarity_search_with_score(
                pergunta,
                k=5
            )
        )

        documentos = [
            doc for doc, distancia in documentos_com_distancia
        ]

        if not documentos:
            return jsonify({
                "resposta": (
                    "Não foram encontrados trechos na base "
                    "de conhecimento."
                ),
                "fontes": []
            })

        contexto, fontes = preparar_contexto(documentos)

        if not contexto.strip():
            return jsonify({
                "resposta": "Os trechos recuperados estão vazios.",
                "fontes": []
            })

        resposta = gerar_resposta(pergunta, contexto)

        return jsonify({
            "resposta": resposta,
            "fontes": fontes
        })

    except Exception as e:

        print("ERRO:", str(e))

        return jsonify({
            "resposta": "Ocorreu um erro ao consultar o sistema: " + str(e),
            "fontes": []
        }), 500


# =====================================================
# 6. INICIALIZAÇÃO
# =====================================================

if __name__ == "__main__":
    print("Assistente CDC iniciado!")
    app.run(host="127.0.0.1", port=5000, debug=False)

