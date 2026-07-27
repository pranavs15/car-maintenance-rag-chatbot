from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from embeddings.embedding_model import get_embedding_model
from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    OLLAMA_MODEL,
    TOP_K
)

# -----------------------------
# Load Embedding Model
# -----------------------------
embedding_model = get_embedding_model()

# -----------------------------
# Load ChromaDB
# -----------------------------
db = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME
)

# -----------------------------
# Better Retriever (MMR)
# -----------------------------
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 70,
        "lambda_mult": 0.85
    }
)

# -----------------------------
# Load Local Gemma
# -----------------------------
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0
)


# -----------------------------
# Remove useless text
# -----------------------------
def clean_context(text):

    unwanted = [
        "copyright",
        "all rights reserved",
        "please read these warnings",
        "you must answer that you have read",
        "version",
        "audi ag",
        "ingolstadt",
        "publisher",
        "table of contents",
        "contents",
        "page",
    ]

    lines = []

    for line in text.split("\n"):

        lower = line.lower().strip()

        if any(word in lower for word in unwanted):
            continue

        if len(lower) < 5:
            continue

        lines.append(line)

    return "\n".join(lines)


# -----------------------------
# Chat Function
# -----------------------------
def ask_question(question):

    docs = db.similarity_search_with_score(question, k=10)

    for doc, score in docs:
        print(score)

    print("\n================ RETRIEVED CHUNKS ================\n")

    cleaned_chunks = []

    for i, doc in enumerate(docs, start=1):

        cleaned = clean_context(doc.page_content)

        cleaned_chunks.append(cleaned)

        print(f"\n----------- Chunk {i} -----------\n")
        print(cleaned[:1200])

    context = "\n\n".join(cleaned_chunks)

    prompt = f"""
You are an expert automotive maintenance assistant.

Your task is to answer ONLY using the provided Context.

Rules:

1. Read ALL context carefully.

2. Ignore:
- Copyright notices
- Warning pages
- Safety disclaimers
- Headers
- Footers
- Page numbers
- Repeated text

3. Focus only on technical repair information.

4. If multiple checks or steps are mentioned,
list every one of them.

5. Never invent information.

6. If the answer cannot be found in the context,
reply exactly:

I don't know.

Provide a concise but complete answer.

======================
Context
======================

{context}

======================
Question
======================

{question}

======================
Answer
======================
"""

    response = llm.invoke(prompt)

    return {
        "question": question,
        "answer": response.content,
        "context": context
    }


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":

    result = ask_question(
        "What checks should be performed before carrying out repairs and fault finding?"
    )

    print("\n================ FINAL ANSWER ================\n")
    print(result["answer"])