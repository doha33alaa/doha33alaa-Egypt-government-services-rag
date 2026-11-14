import json
import math
import re
import time
import hashlib
import requests
import chromadb

from collections import Counter
from pathlib import Path


# ==========================================
# 1. Project Configuration
# ==========================================

OLLAMA_URL = "http://localhost:11434"

CHAT_MODEL = "llama3.2:1b"
EMBED_MODEL = "bge-m3"

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

SERVICES_FILE = DATA_DIR / "egypt_services_rag_organized.json"

# Optional now: each service already carries its own "service_description".
# The separate file is used only for services that have no description.
DESCRIPTIONS_FILE = DATA_DIR / "service_descriptions.json"

COLLECTION_NAME = "egypt_gov_services"

# Changing this number forces an automatic rebuild of the database.
INDEX_VERSION = "service-level-v2"

CHUNK_SIZE = 700       # kept for compatibility, no longer used
CHUNK_OVERLAP = 100    # kept for compatibility, no longer used

TOP_K = 4              # how many ranked results are returned
CANDIDATE_K = 12       # how many semantic candidates are re-ranked
BATCH_SIZE = 50

# Hybrid ranking weights (semantic similarity + service-name match)
SEMANTIC_WEIGHT = 0.6
LEXICAL_WEIGHT = 0.4

# If the best result is weaker than this semantically AND shares no
# meaningful word with any service name, we say "not found".
# Tune it after looking at the scores printed for real questions.
MIN_SEMANTIC_SIM = 0.30


# ==========================================
# 2. Generate Embeddings
# ==========================================

def get_embedding(text):

    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text
        },
        timeout=180
    )

    response.raise_for_status()

    result = response.json()

    if "embedding" not in result:
        raise ValueError("Embedding was not returned by Ollama.")

    return result["embedding"]


# ==========================================
# 3. Arabic Text Helpers
# ==========================================

_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")

# Words that describe the *question*, not the service.
# They must not influence which service is selected.
_RAW_STOPWORDS = """
ازاي إزاي كيف ايه إيه ما ماذا هي هو هل عايز عايزه عايزة عاوز عاوزة
اريد أريد محتاج محتاجة ممكن لو انا أنا اللي الي في علي على من عن الى إلى
مع او أو ان أن هذا هذه ده دي دا كده بتاع بتاعة
خدمة الخدمة خدمه الخدمه
المستندات مستندات الأوراق الاوراق اوراق ورق المطلوبة المطلوب
الخطوات خطوات طريقة طريقه
الرسوم رسوم التكلفة تكلفة بكام كام
الأماكن الاماكن اماكن المكان فين أين اين
اعمل اعمله اقدم أقدم اروح أروح
يعني معني معنى شرح تعريف عرف
"""


def normalize_ar(text):

    text = str(text or "").lower()
    text = _DIACRITICS.sub("", text)

    for old in "أإآٱ":
        text = text.replace(old, "ا")

    text = text.replace("ى", "ي").replace("ة", "ه")

    text = re.sub(r"[^\w\s]", " ", text)

    return text


STOPWORDS = {
    normalize_ar(word)
    for word in _RAW_STOPWORDS.split()
}


def tokenize(text):

    tokens = []

    for token in normalize_ar(text).split():

        if token in STOPWORDS:
            continue

        if token.startswith("ال") and len(token) > 4:
            token = token[2:]
        elif (
            token[:3] in ("وال", "بال", "فال", "كال")
            and len(token) > 5
        ):
            token = token[3:]

        if len(token) < 2 or token in STOPWORDS:
            continue

        tokens.append(token)

    return tokens


# ==========================================
# 4. Prepare Service Data
# ==========================================

def format_value(value):

    if value is None:
        return "غير موثق"

    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


def load_descriptions():

    if not DESCRIPTIONS_FILE.exists():
        return {}

    with DESCRIPTIONS_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        descriptions = json.load(file)

    if not isinstance(descriptions, dict):
        raise ValueError(
            "Descriptions JSON must contain an object."
        )

    return descriptions


def get_description(service, descriptions):
    """The description stored inside the service record always wins.

    The separate descriptions file is keyed by service_id, so it is only
    a fallback, and only safe if the ids were never renumbered.
    """

    own = service.get("service_description")

    if isinstance(own, str) and own.strip():
        return own.strip()

    record = descriptions.get(
        str(service.get("service_id", 0)),
        {}
    )

    if isinstance(record, dict):
        return str(
            record.get("service_description", "")
        ).strip()

    return ""


def short_category(category):
    """Drop the ordinal prefix, e.g. 'أولًا: خدمات X' -> 'خدمات X'."""

    return str(category).split(":")[-1].strip()


def build_search_text(service_name, description, category):
    """The ONLY text that is embedded: name + description + category.

    Documents, steps, fees and locations are deliberately left out, so a
    service is not matched just because its documents mention a topic.
    """

    parts = [f"اسم الخدمة: {service_name}"]

    if description:
        parts.append(f"شرح الخدمة: {description}")

    parts.append(f"التصنيف: {short_category(category)}")

    return "\n".join(parts)


def build_service_context(service, description):
    """Full service details, used as context for the language model."""

    return f"""
اسم الخدمة: {service.get("service_name", "")}

التصنيف: {service.get("category", "")}

شرح الخدمة: {description or "غير متوفر"}

المستندات المطلوبة:
{format_value(service.get("required_documents") or [])}

الخطوات:
{format_value(service.get("steps"))}

الرسوم:
{format_value(service.get("fees"))}

أماكن تقديم الخدمة:
{format_value(service.get("locations") or [])}

المصادر الرسمية:
{format_value(service.get("official_sources") or [])}
""".strip()


def prepare_service(service, descriptions):

    service_id = service.get("service_id", 0)

    service_name = str(
        service.get(
            "service_name",
            "خدمة حكومية غير محددة"
        )
    )

    category = str(
        service.get(
            "category",
            "غير محدد"
        )
    )

    description = get_description(service, descriptions)

    official_sources = service.get("official_sources") or []

    return {
        "text": build_search_text(
            service_name,
            description,
            category
        ),
        "source": SERVICES_FILE.name,
        "service_id": int(service_id),
        "service_name": service_name,
        "description": description,
        "category": category,
        "verification_status": str(
            service.get(
                "verification_status",
                "غير محددة"
            )
        ),
        "official_sources": " | ".join(
            str(url) for url in official_sources
        )
    }


# ==========================================
# 5. Load Documents
# ==========================================

def load_documents():

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data folder not found: {DATA_DIR}"
        )

    if not SERVICES_FILE.exists():
        raise FileNotFoundError(
            f"Services file not found: {SERVICES_FILE}"
        )

    with SERVICES_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    services = data.get("services", [])

    if not isinstance(services, list):
        raise ValueError(
            "'services' must be a list."
        )

    descriptions = load_descriptions()

    print(
        f"Read {len(services)} services from "
        f"{SERVICES_FILE.name}"
    )

    documents = []
    seen_ids = set()

    for service in services:

        if not isinstance(service, dict):
            continue

        document = prepare_service(
            service,
            descriptions
        )

        if document["service_id"] in seen_ids:
            raise ValueError(
                f"Duplicate service_id: {document['service_id']}"
            )

        seen_ids.add(document["service_id"])

        if not document["description"]:
            print(
                f"Warning: no description for service "
                f"{document['service_id']} - {document['service_name']}"
            )

        documents.append(document)

    if not documents:
        raise ValueError("No documents were loaded.")

    return documents


# ==========================================
# 6. Split Text into Chunks (kept, unused)
# ==========================================

def split_into_chunks(text):

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ==========================================
# 7. Create Dataset Fingerprint
# ==========================================

def get_dataset_fingerprint():

    hasher = hashlib.sha256()

    if not SERVICES_FILE.exists():
        raise FileNotFoundError(SERVICES_FILE)

    hasher.update(SERVICES_FILE.read_bytes())

    if DESCRIPTIONS_FILE.exists():
        hasher.update(DESCRIPTIONS_FILE.read_bytes())

    hasher.update(EMBED_MODEL.encode("utf-8"))
    hasher.update(INDEX_VERSION.encode("utf-8"))

    return hasher.hexdigest()


# ==========================================
# 8. Build or Load Vector Database
# ==========================================

def build_database():

    print("\nBuilding the knowledge base...")

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    fingerprint = get_dataset_fingerprint()

    collection = None

    try:

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

    except Exception:

        collection = None

    if collection is not None:

        saved_fingerprint = (
            collection.metadata or {}
        ).get("dataset_fingerprint")

        if (
            saved_fingerprint == fingerprint
            and collection.count() > 0
        ):

            print("\nExisting database found.")
            print(f"Embedding model: {EMBED_MODEL}")
            print(f"Stored services: {collection.count()}")
            print("No rebuilding is needed.")

            return collection

        print("\nData or embedding settings have changed.")
        print("Rebuilding the database...")

        client.delete_collection(
            name=COLLECTION_NAME
        )

    else:

        print("\nNo existing database found.")
        print("Creating a new database...")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBED_MODEL,
            "dataset_fingerprint": fingerprint
        }
    )

    documents = load_documents()

    all_texts = []
    all_embeddings = []
    all_metadata = []
    all_ids = []

    print("\nGenerating one embedding per service...")

    start_time = time.time()

    for document in documents:

        embedding = get_embedding(document["text"])

        all_texts.append(document["text"])
        all_embeddings.append(embedding)

        all_metadata.append({
            "source": str(document["source"]),
            "service_id": int(document["service_id"]),
            "service_name": str(document["service_name"]),
            "description": str(document["description"]),
            "category": str(document["category"]),
            "verification_status": str(
                document["verification_status"]
            ),
            "official_sources": str(
                document["official_sources"]
            )
        })

        all_ids.append(f"service_{document['service_id']}")

        print(f"Processed: {document['service_name']}")

    print(
        f"\nEmbedding time: "
        f"{time.time() - start_time:.2f} seconds"
    )

    print("\nSaving embeddings to ChromaDB...")

    for start in range(0, len(all_texts), BATCH_SIZE):

        end = start + BATCH_SIZE

        collection.add(
            ids=all_ids[start:end],
            documents=all_texts[start:end],
            embeddings=all_embeddings[start:end],
            metadatas=all_metadata[start:end]
        )

    print("\nDatabase built successfully!")
    print(f"Stored {collection.count()} services in ChromaDB.")

    return collection


# ==========================================
# 9. Hybrid Search (semantic + name match)
# ==========================================

def build_lexical_index(metadatas):

    docs = {}

    for metadata in metadatas:

        service_id = metadata.get("service_id")

        name_tokens = set(
            tokenize(metadata.get("service_name", ""))
        )

        description_tokens = set(
            tokenize(metadata.get("description", ""))
        )

        docs[service_id] = (name_tokens, description_tokens)

    total = len(docs)

    document_frequency = Counter()

    for name_tokens, description_tokens in docs.values():

        for token in name_tokens | description_tokens:
            document_frequency[token] += 1

    # Rare words (e.g. "الاجتماعية") weigh more than common ones
    # (e.g. "بطاقة"), so a service is not chosen for a shared word.
    idf = {
        token: math.log((total + 1) / (count + 0.5))
        for token, count in document_frequency.items()
    }

    return docs, idf


def lexical_score(query_tokens, service_id, docs, idf):

    known = [
        token for token in set(query_tokens)
        if token in idf
    ]

    if not known or service_id not in docs:
        return 0.0

    total = sum(idf[token] for token in known)

    if total <= 0:
        return 0.0

    name_tokens, description_tokens = docs[service_id]

    name_hit = sum(
        idf[token] for token in known
        if token in name_tokens
    )

    description_hit = sum(
        idf[token] for token in known
        if token in description_tokens
    )

    score = (
        0.75 * name_hit / total
        + 0.25 * description_hit / total
    )

    return min(1.0, score)


def search_services(collection, question, top_k=TOP_K):

    if collection.count() == 0:
        return []

    question_embedding = get_embedding(question)

    number_of_results = min(
        CANDIDATE_K,
        collection.count()
    )

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    all_metadata = collection.get(
        include=["metadatas"]
    )["metadatas"]

    docs, idf = build_lexical_index(all_metadata)

    query_tokens = tokenize(question)

    ranked = []

    for index, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][index]
        distance = results["distances"][0][index]

        semantic = max(0.0, min(1.0, 1.0 - distance))

        lexical = lexical_score(
            query_tokens,
            metadata.get("service_id"),
            docs,
            idf
        )

        final_score = (
            SEMANTIC_WEIGHT * semantic
            + LEXICAL_WEIGHT * lexical
        )

        ranked.append({
            "document": document,
            "metadata": metadata,
            "distance": distance,
            "semantic": semantic,
            "lexical": lexical,
            "score": final_score
        })

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return ranked[:top_k]


# ==========================================
# 10. Retrieve Original JSON Service
# ==========================================

def get_service_record(service_id):

    with SERVICES_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    services = data.get("services", [])

    for service in services:

        if service.get("service_id") == service_id:
            return service

    return None


# ==========================================
# 11. Format Database Values
# ==========================================

def format_answer_value(value):

    if value is None or value == "":
        return None

    if isinstance(value, list):

        if not value:
            return None

        return "\n".join(
            f"- {item}" for item in value
        )

    if isinstance(value, dict):

        if not value:
            return None

        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2
        )

    return str(value)


# ==========================================
# 12. Answer Structured Questions Directly
# ==========================================

def answer_from_database(service, question):

    question_lower = question.lower()

    asks_documents = any(word in question_lower for word in [
        "المستندات",
        "مستندات",
        "الأوراق",
        "الاوراق",
        "المطلوب",
        "ورق"
    ])

    asks_steps = any(word in question_lower for word in [
        "الخطوات",
        "خطوات",
        "ازاي",
        "إزاي",
        "كيف",
        "طريقة"
    ])

    asks_fees = any(word in question_lower for word in [
        "الرسوم",
        "رسوم",
        "التكلفة",
        "تكلفة",
        "بكام",
        "كام"
    ])

    asks_locations = any(word in question_lower for word in [
        "الأماكن",
        "الاماكن",
        "المكان",
        "فين",
        "أين",
        "اماكن"
    ])

    if not any([
        asks_documents,
        asks_steps,
        asks_fees,
        asks_locations
    ]):

        return None

    answer_parts = []

    if asks_documents:

        documents = format_answer_value(
            service.get("required_documents")
        )

        if documents:

            answer_parts.append(
                "المستندات المذكورة في قاعدة البيانات:\n"
                + documents
            )

        else:

            answer_parts.append(
                "المستندات المطلوبة غير موثقة في قاعدة البيانات."
            )

    if asks_steps:

        steps = format_answer_value(
            service.get("steps")
        )

        if steps:

            answer_parts.append(
                "خطوات الخدمة:\n" + steps
            )

        else:

            answer_parts.append(
                "خطوات الخدمة غير موثقة في قاعدة البيانات."
            )

    if asks_fees:

        fees = format_answer_value(
            service.get("fees")
        )

        if fees:

            answer_parts.append(
                "رسوم الخدمة:\n" + fees
            )

        else:

            answer_parts.append(
                "رسوم الخدمة غير موثقة في قاعدة البيانات."
            )

    if asks_locations:

        locations = format_answer_value(
            service.get("locations")
        )

        if locations:

            answer_parts.append(
                "أماكن تقديم الخدمة:\n" + locations
            )

        else:

            answer_parts.append(
                "أماكن تقديم الخدمة غير موثقة في قاعدة البيانات."
            )

    verification_status = service.get(
        "verification_status",
        ""
    )

    if verification_status == "needs_official_verification":

        answer_parts.append(
            "\nتنبيه: المعلومات المذكورة موجودة في قاعدة البيانات، "
            "لكنها تحتاج إلى مراجعة المصدر الحكومي الرسمي قبل اعتمادها."
        )

    return "\n\n".join(answer_parts)


# ==========================================
# 13. Answer "What is this service?" Directly
# ==========================================

_DEFINITION_PHRASES = [
    "يعني ايه",
    "معني",
    "شرح",
    "تعريف",
    "ما هي خدمه",
    "ما هو",
    "ايه هي",
    "ايه هو",
    "عرفني",
    "وضحلي",
    "اشرح",
]


def answer_definition(service, description, question):
    """Return the stored description for 'what does X mean?' questions.

    A small model (llama3.2:1b) writes poor Arabic, so for definition
    questions we return the curated description as it is.
    """

    if not description:
        return None

    normalized = normalize_ar(question)

    if not any(
        phrase in normalized
        for phrase in _DEFINITION_PHRASES
    ):
        return None

    return (
        f"{service.get('service_name', '')}:\n{description}"
    )


# ==========================================
# 14. Ask a Question
# ==========================================

def ask_question(collection, question):

    if not question or not question.strip():

        return (
            "من فضلك اكتب سؤالك أولًا.",
            []
        )

    question = question.strip()

    # Step 1: Search the knowledge base (hybrid ranking)

    print("\n[1] Generating question embedding...")

    start_time = time.time()

    results = search_services(
        collection,
        question,
        top_k=TOP_K
    )

    print(
        f"Retrieval completed in "
        f"{time.time() - start_time:.2f} seconds"
    )

    if not results:

        return (
            "لم أجد معلومات مرتبطة بسؤالك في قاعدة البيانات.",
            []
        )

    # Step 2: Display retrieved results

    print("\nRetrieved results:")

    for index, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        print(f"\nResult {index}")

        print(
            "Service:",
            metadata.get("service_name")
        )

        print(
            "Distance:",
            round(result["distance"], 4),
            "| Semantic:",
            round(result["semantic"], 3),
            "| Name match:",
            round(result["lexical"], 3),
            "| Final score:",
            round(result["score"], 3)
        )

    # Step 3: Reject questions that match nothing

    best_result = results[0]

    if (
        best_result["semantic"] < MIN_SEMANTIC_SIM
        and best_result["lexical"] == 0
    ):

        return (
            "لم أجد خدمة مطابقة لسؤالك في قاعدة البيانات. "
            "حاول كتابة اسم الخدمة بشكل أوضح.",
            []
        )

    best_metadata = best_result["metadata"]

    selected_service_id = best_metadata.get("service_id")

    print("\nSelected service:")
    print(best_metadata.get("service_name"))

    # Step 4: Retrieve the original JSON record

    service = get_service_record(
        selected_service_id
    )

    if service is None:

        return (
            "لم أجد سجل الخدمة الأصلي في ملف البيانات.",
            []
        )

    description = (
        best_metadata.get("description")
        or get_description(service, load_descriptions())
    )

    # Step 5: Prepare source information

    source_info = {
        "service_id": selected_service_id,
        "service_name": best_metadata.get("service_name"),
        "category": best_metadata.get("category"),
        "source": best_metadata.get("source"),
        "verification_status": best_metadata.get(
            "verification_status"
        ),
        "official_sources": best_metadata.get(
            "official_sources"
        )
    }

    sources = [source_info]

    # Step 6: Answer structured questions directly from JSON

    database_answer = answer_from_database(
        service,
        question
    )

    if database_answer is not None:

        print(
            "\nAnswer generated directly from the JSON database."
        )

        return database_answer, sources

    # Step 7: Answer "what does this service mean?" directly

    definition_answer = answer_definition(
        service,
        description,
        question
    )

    if definition_answer is not None:

        print(
            "\nAnswer generated directly from the service description."
        )

        return definition_answer, sources

    # Step 8: Build the context from the selected service only

    context = build_service_context(
        service,
        description
    )

    print("\nContext prepared from the selected service record.")

    # Step 9: Prepare the prompt

    prompt = f"""
أنت مساعد ذكي متخصص في شرح الخدمات الحكومية المصرية.

مهمتك هي مساعدة المواطن على فهم الخدمة الحكومية
باستخدام المعلومات الموجودة في السياق فقط.

اسم الخدمة:
{best_metadata.get("service_name")}

السياق والمعلومات المتاحة:
{context}

سؤال المستخدم:
{question}

تعليمات مهمة جدًا:

1. أجب باللغة العربية الواضحة والبسيطة.
2. افهم المقصود من السؤال قبل كتابة الإجابة.
3. إذا كان السؤال يسأل عن ماهية الخدمة أو تعريفها،
   اشرح الغرض من الخدمة وما الذي تساعد المواطن على إنجازه.
4. استخدم فقرة من جملتين إلى أربع جمل عند شرح الخدمة.
5. لا تكتفِ بتكرار اسم الخدمة في الإجابة.
6. استخدم حقل شرح الخدمة والمعلومات المتاحة في السياق.
7. لا تخترع معلومات أو مستندات أو رسوم أو خطوات.
8. إذا كانت المعلومة غير موجودة، قل إنها غير موثقة.
9. لا تخلط بين معلومات الخدمات المختلفة.
10. حالة needs_official_verification تعني أن المعلومات
    تحتاج إلى مراجعة المصدر الرسمي، ولا تعني أنها غير موجودة.
11. لا ترفض الإجابة إذا كانت المعلومات موجودة في السياق.
12. لا تقدم معلومات حكومية من خارج السياق.

اكتب إجابة مفيدة ومباشرة تجيب عن سؤال المواطن.
لا تكرر اسم الخدمة فقط.

الإجابة:
"""

    # Step 10: Generate answer

    print("\n[2] Generating answer with Ollama...")

    generation_start = time.time()

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=180
    )

    response.raise_for_status()

    result = response.json()

    answer = result.get(
        "response",
        ""
    ).strip()

    print(
        f"Generation completed in "
        f"{time.time() - generation_start:.2f} seconds"
    )

    if not answer:

        answer = (
            "لم يتمكن النموذج من إنشاء إجابة. "
            "يرجى المحاولة مرة أخرى."
        )

    return answer, sources