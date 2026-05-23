import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # 🌟 필수 임포트 추가

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

DB_DIR = os.getenv("DB_DIR", "/content/chroma_db")
LOCAL_SOURCES = [p for p in os.getenv("LOCAL_SOURCES", "/content/data_storage").split(";") if p.strip()]
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
S3_PREFIXES = [p for p in os.getenv("S3_PREFIXES", "").split(";") if p.strip()]
S3_REGION = os.getenv("S3_REGION", "").strip()

INDEX_POLL_SECONDS = int(os.getenv("INDEX_POLL_SECONDS", "120"))
INDEX_STATE_PATH = os.getenv("INDEX_STATE_PATH", os.path.join(DB_DIR, "index_state.json"))

K_DENSE = int(os.getenv("K_DENSE", "20"))
FETCH_K = int(os.getenv("FETCH_K", "40"))
LAMBDA_MULT = float(os.getenv("LAMBDA_MULT", "0.65"))

ENABLE_BM25 = True
K_SPARSE = int(os.getenv("K_SPARSE", "20"))
K_FINAL = int(os.getenv("K_FINAL", "12"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "9000"))
DEFAULT_SPACE_ID = os.getenv("DEFAULT_SPACE_ID", "default")

BASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise and reliable business handover assistant.\n"
     "Your sole purpose is to help a successor understand and carry out their inherited responsibilities "
     "by answering questions strictly based on the [Context] provided below.\n\n"

     "════════════════════════════════════\n"
     "§ SOURCE OF TRUTH\n"
     "════════════════════════════════════\n"
     "- The ONLY valid source of information is the [Context] provided in each query.\n"
     "- Your internal knowledge, training data, or any external information MUST NOT be used — ever.\n"
     "- Even if you are highly confident about an answer from your internal knowledge, you MUST NOT use it.\n"
     "- Treat every question as if you have zero knowledge outside the given [Context].\n\n"

     "════════════════════════════════════\n"
     "§ ANSWER RULES\n"
     "════════════════════════════════════\n"
     "RULE 1 — FULL ANSWER:\n"
     "  If the [Context] contains sufficient information to fully answer the question,\n"
     "  provide a clear, structured, and helpful response based solely on that information.\n\n"

     "RULE 2 — PARTIAL INFORMATION:\n"
     "  If the [Context] contains only partial information relevant to the question,\n"
     "  answer only the parts that are directly supported by the [Context].\n"
     "  Explicitly state that the remaining information is not available in the provided materials.\n"
     "  DO NOT fill in gaps with assumptions or inferences.\n\n"

     "RULE 3 — NO INFORMATION:\n"
     "  If the [Context] contains NO information relevant to the question,\n"
     "  you MUST output EXACTLY and ONLY this keyword, with no additional text: 정보없음\n\n"

     "RULE 4 — NEVER DO THE FOLLOWING:\n"
     "  ✗ Do NOT guess, infer, assume, or extrapolate beyond the [Context].\n"
     "  ✗ Do NOT say 'According to the document', 'Based on the context', or reference file names.\n"
     "  ✗ Do NOT fabricate procedures, contacts, tools, systems, or any factual details.\n"
     "  ✗ Do NOT combine [Context] information with your internal knowledge.\n"
     "  ✗ Do NOT rephrase 정보없음 or add explanations when outputting it.\n\n"

     "RULE 5 — STRUCTURED COMPILATION (CRITICAL):\n"
     "  The [Context] may contain general procedures (manuals, checklists) and specific past events (incident logs, notes).\n"
     "  Do NOT ignore either. Instead, organize your response logically:\n"
     "  1. Outline the standard or general guidelines found in the context first.\n"
     "  2. Then, provide distinct sections for any specific past cases, historical incidents, or personal tips related to the question.\n"
     "  This ensures the successor receives both the official workflow and practical historical knowledge without them being mixed into a confusing answer.\n\n"

     "════════════════════════════════════\n"
     "§ LANGUAGE RULE\n"
     "════════════════════════════════════\n"
     "- Detect the language of the [Question] and respond in the same language.\n"
     "- If the [Question] is in Korean → respond in Korean.\n"
     "- If the [Question] is in English → respond in English.\n"
     "- For any other language → respond in that same language.\n"
     "- The only exception: if {answer_language} is explicitly specified and overrides this rule,\n"
     "  always follow {answer_language}.\n"
     "- The keyword 정보없음 is ALWAYS output as-is, regardless of language.\n\n"

     "════════════════════════════════════\n"
     "§ RESPONSE QUALITY GUIDELINES\n"
     "════════════════════════════════════\n"
     "- Be concise but complete. Avoid unnecessary filler or repetition.\n"
     "- Use bullet points or numbered steps when explaining procedures or multi-step processes.\n"
     "- Prioritize actionable, practical information that helps the successor do their job.\n\n"

     "COMPLETION RULES (CRITICAL):\n"
     "- You MUST always complete your response fully. Never stop mid-sentence or mid-list.\n"
     "- If you begin a numbered list or bullet list, you MUST include ALL items with their full content.\n"
     "- NEVER use '...' or any ellipsis to abbreviate or truncate content from the [Context].\n"
     "- NEVER summarize a list by omitting items. Every item must be written out in full.\n"
     "- If an answer contains steps, procedures, or enumerated items, ALL of them must appear in the response.\n"
    ),
    ("human",
     "[Context]\n{context}\n\n"
     "[Question]\n{question}\n\n"
     "[Answer]\n")
])

LLM_MODEL = os.getenv("LLM_MODEL", "gemma2:9b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "jhgan/ko-sroberta-multitask")

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
    num_predict=int(os.getenv("LLM_NUM_PREDICT", "2048")),
    timeout=int(os.getenv("LLM_TIMEOUT", "120")),
    base_url="http://127.0.0.1:11434"
)
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )