# agentic_rag.py
# ------------------------------------------------------------
# Agentic RAG demo:
# - Reads a resume PDF
# - Chunks + embeds into Chroma (SBERT)
# - Planner (LLM) selects tools and composes multi-step plans
# - Critic checks grounding (citations) and triggers refinement
# - Policy decides when to finish
# ------------------------------------------------------------

import os
import json
import requests
from dataclasses import dataclass
from typing import List, Dict, Any

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer


# ======== CONFIG ========
PDF_PATH = "./Vivek_B_Resume_DE2.pdf"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_collection"
# You can also use another chat-capable model name supported by OPENAI
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")


# ======== DATA STRUCTURES ========
@dataclass
class RetrievedChunk:
    id: str
    text: str
    page: int
    idx: int


@dataclass
class GoalPolicy:
    require_citations: bool = True
    min_critic_confidence: float = 0.6
    max_steps: int = 6


# ======== PDF READING & CHUNKING ========
def pdf_reader_with_pages(pdf_path: str) -> List[tuple]:
    """Return list of (page_num, text) for pages that have text."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        if t.strip():
            pages.append((i, t))
    return pages


def textsplitter_with_meta(pages: List[tuple]):
    """Return ids, docs, metas ready for Chroma upsert."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs, metas, ids = [], [], []
    k = 0
    for page_num, text in pages:
        for d in splitter.create_documents([text]):
            docs.append(d.page_content)
            metas.append({"page": page_num, "idx": k})
            ids.append(f"p{page_num}_c{k}")
            k += 1
    return ids, docs, metas


# ======== VECTOR DB (CHROMA) ========
def init_vector_db():
    client = chromadb.PersistentClient(path=DB_PATH)

    # Local sentence-transformer embedder
    model = SentenceTransformer("all-MiniLM-L6-v2")

    class SBERTEmbeddingFunction:
        def __call__(self, input):
            if isinstance(input, str):
                input = [input]
            return model.encode(input).tolist()

        def name(self):
            return "sbert-mini"

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=SBERTEmbeddingFunction()
    )


def upsert_documents(collection, ids, docs, metas, force_rebuild: bool = False):
    if force_rebuild and collection.count() > 0:
        collection.delete(where={})
    if ids:
        # upsert ensures the collection mirrors current content
        collection.upsert(ids=ids, documents=docs, metadatas=metas)


def retrieve(collection, query: str, n_results: int = 4) -> List[RetrievedChunk]:
    res = collection.query(query_texts=[query], n_results=n_results)
    chunks: List[RetrievedChunk] = []
    # Handle case with no results
    if not res or not res.get("documents") or not res["documents"][0]:
        return chunks
    for doc, mid, meta in zip(res["documents"][0], res["ids"][0], res["metadatas"][0]):
        chunks.append(RetrievedChunk(id=mid, text=doc, page=meta["page"], idx=meta["idx"]))
    return chunks


# ======== LLM CLIENT (Together Chat Completions) ========
import os, requests, json

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def chat_openai(messages, model="gpt-4o-mini", temperature=0.2, max_tokens=768):
    """
    OpenAI Chat Completions call with robust error logging.
    `messages` must be a list of {"role": "...", "content": "..."}.
    """
    try:
        r = requests.post(
            "https://api.together.xyz/v1/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        if r.status_code != 200:
            # Print the server's error details so you can see the real cause
            try:
                err = r.json()
            except Exception:
                err = {"raw_text": r.text}
            return f"[Error] OpenAI API request failed ({r.status_code}): {json.dumps(err, ensure_ascii=False)[:1200]}"
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error] OpenAI API request failed: {e}"


def json_from_model(prompt_sys, prompt_user):
    messages = [
        {"role": "system", "content": prompt_sys},
        {"role": "user", "content": prompt_user}
    ]
    text = chat_openai(messages)
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
        # fallback if still broken
        return {"tool": "qa", "args": {}, "confidence": 0.0, "thoughts": f"Bad JSON: {text[:200]}..."}

# ======== CITATIONS UTILS ========
def ids_to_brackets(chunks: List[RetrievedChunk]) -> str:
    return " ".join([f"[p{c.page}:c{c.idx}]" for c in chunks])


# ======== TOOLS ========
# All RAG tools take retrieved chunks and return dict with "output", "citations", "confidence"

def tool_answer_question(context_chunks: List[RetrievedChunk], question: str) -> Dict[str, Any]:
    context_text = "\n\n".join([c.text for c in context_chunks]) if context_chunks else ""
    citations_hint = ids_to_brackets(context_chunks)
    prompt = f"""Use ONLY the context to answer. If not supported, say "I don't know".
Cite chunk ids inline like [pX:cY].

Context:
{context_text}

Question: {question}

Answer (with citations like {citations_hint}):
"""
    out = chat_openai([
        {"role": "system", "content": "You are a precise assistant. Ground every claim in the context and cite [pX:cY]."},
        {"role": "user", "content": prompt}
    ])
    return {"output": out, "citations": [c.id for c in context_chunks], "confidence": 0.6}


def tool_summarize(context_chunks: List[RetrievedChunk]) -> Dict[str, Any]:
    context_text = "\n\n".join([c.text for c in context_chunks]) if context_chunks else ""
    citations_hint = ids_to_brackets(context_chunks)
    prompt = f"Summarize the resume in 5–7 bullets. Cite chunks like [pX:cY].\n\n{context_text}\n\n(Use citations like {citations_hint})"
    out = chat_openai([
        {"role": "system", "content": "Concise, factual, cite chunks with [pX:cY]."},
        {"role": "user", "content": prompt}
    ])
    return {"output": out, "citations": [c.id for c in context_chunks], "confidence": 0.7}


def tool_extract_skills(context_chunks: List[RetrievedChunk]) -> Dict[str, Any]:
    context_text = "\n\n".join([c.text for c in context_chunks]) if context_chunks else ""
    citations_hint = ids_to_brackets(context_chunks)
    prompt = f"""List ONLY technical skills present in the text.
Group into: Languages, Data/Big Data, Cloud, Orchestration, DevOps/Other.
Cite chunks like [pX:cY].

{context_text}

(Use citations like {citations_hint})
"""
    out = chat_openai([
        {"role": "system", "content": "Extract skills strictly from context, grouped, with citations."},
        {"role": "user", "content": prompt}
    ])
    return {"output": out, "citations": [c.id for c in context_chunks], "confidence": 0.7}


def tool_compare_with_jd(context_chunks: List[RetrievedChunk], jd_text: str) -> Dict[str, Any]:
    context_text = "\n\n".join([c.text for c in context_chunks]) if context_chunks else ""
    citations_hint = ids_to_brackets(context_chunks)
    prompt = f"""Compare resume vs JD. Output:
- Fit score (0-100)
- Matching skills (with citations)
- Gaps
- 3 bullets to improve

Resume (context):
{context_text}

Job Description:
{jd_text}

(Use citations like {citations_hint})
"""
    out = chat_openai([
        {"role": "system", "content": "Ground matches in resume with [pX:cY]. No hallucinations."},
        {"role": "user", "content": prompt}
    ])
    return {"output": out, "citations": [c.id for c in context_chunks], "confidence": 0.65}


# ---- Non-RAG helpers / extensibility ----
def tool_refine_query(question: str, last_answer: str) -> List[str]:
    sys = "Return only valid JSON."
    user = f"""Given the question and last answer, propose 3 refined retrieval queries to improve recall.
Return as JSON: {{"queries": ["...", "...", "..."]}}

Question: {question}
Last answer: {last_answer}
"""
    obj = json_from_model(sys, user)
    queries = obj.get("queries") or obj.get("args", {}).get("queries") or []
    # Ensure a list of strings
    queries = [q for q in queries if isinstance(q, str)]
    if not queries:
        queries = [question]  # fallback
    return queries[:3]


def tool_search_jobs(_: Dict[str, Any]) -> Dict[str, Any]:
    # Stub for integrating a jobs API
    return {"output": "Stub: Company A (Data Engineer), Company B (ML Engineer)", "citations": [], "confidence": 0.5}


def tool_write_email_to_file(subject: str, body: str, path: str = "./output_email.txt") -> Dict[str, Any]:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Subject: {subject}\n\n{body}\n")
        return {"output": f"Email written to {path}", "citations": [], "confidence": 0.95}
    except Exception as e:
        return {"output": f"[Error] Writing email failed: {e}", "citations": [], "confidence": 0.1}


def tool_find_calendar_slots(_: Dict[str, Any]) -> Dict[str, Any]:
    # Stub for integrating Calendar
    return {"output": "Stub: Available slots Tue 10-11, Wed 2-3", "citations": [], "confidence": 0.5}


TOOLS = {
    "qa": tool_answer_question,
    "summarize": tool_summarize,
    "skills": tool_extract_skills,
    "compare_with_jd": tool_compare_with_jd,
    # non-RAG / integrations
    "search_jobs": tool_search_jobs,
    "write_email": tool_write_email_to_file,
    "find_slots": tool_find_calendar_slots,
}


# ======== PLANNER ========
PLANNER_SYS = """You are an agentic planner for a RAG system.
Available tools: qa, summarize, skills, compare_with_jd, search_jobs, write_email, find_slots.
You MUST output valid JSON exactly like:
{
  "thoughts": "brief reasoning",
  "tool": "<one of: qa|summarize|skills|compare_with_jd|search_jobs|write_email|find_slots|finish>",
  "args": { ... },  // tool arguments or {"final": "..."} if tool == "finish"
  "confidence": 0.0
}
Rules:
- Prefer grounded RAG tools first (qa/summarize/skills/compare_with_jd).
- If the task is complete, set tool="finish" and put the final answer in args.final.
- Use confidence in [0.0, 1.0].
- Return JSON ONLY, no extra text.
- You MUST return valid JSON only. Do not include explanations, markdown, or extra text.

"""


def plan(user_goal: str, scratchpad: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Only keep last few observations to stay within context window
    recent = scratchpad[-5:] if len(scratchpad) > 5 else scratchpad
    user = f"""Goal: {user_goal}

Scratchpad (recent observations):
{json.dumps(recent, ensure_ascii=False, indent=2)}

Return JSON only."""
    obj = json_from_model(PLANNER_SYS, user)

    # Guard/normalize
    tool = obj.get("tool", "qa")
    if tool not in ("qa", "summarize", "skills", "compare_with_jd", "search_jobs", "write_email", "find_slots", "finish"):
        tool = "qa"
    args = obj.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    conf = obj.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0

    return {"tool": tool, "args": args, "confidence": conf, "thoughts": obj.get("thoughts", "")}


# ======== CRITIC (SELF-CHECK) ========
def critic_grounding(answer_text: str, used_chunk_ids: List[str]) -> Dict[str, Any]:
    sys = "Return only JSON."
    user = f"""Given the answer, check grounding against this set of chunk ids: {used_chunk_ids}.
Map each claim to supporting chunk IDs if present.
Return JSON:
{{
  "supported": [{{"claim": "...","chunks": ["pX:cY", "..."]}}, ...],
  "unsupported": ["claim1", "claim2"],
  "confidence": 0.0
}}

Answer:
{answer_text}
"""
    obj = json_from_model(sys, user)
    # Normalize fields
    obj.setdefault("supported", [])
    obj.setdefault("unsupported", [])
    try:
        obj["confidence"] = float(obj.get("confidence", 0.0))
    except Exception:
        obj["confidence"] = 0.0
    return obj


def goal_met(candidate_final: str, critic_report: Dict[str, Any], policy: GoalPolicy) -> bool:
    if policy.require_citations and "[p" not in candidate_final:
        return False
    if critic_report.get("confidence", 0.0) < policy.min_critic_confidence:
        return False
    return True


# ======== AGENT LOOP ========
def agent_loop(user_goal: str, collection, policy: GoalPolicy = GoalPolicy()) -> str:
    scratchpad: List[Dict[str, Any]] = []
    last_answer = ""

    # Initial retrieval for the goal
    retrieved = retrieve(collection, user_goal, n_results=4)
    if not retrieved:
        # Try a generic query to get something to start with
        retrieved = retrieve(collection, "resume summary experience skills projects", n_results=4)

    for step in range(policy.max_steps):
        plan_json = plan(user_goal, scratchpad)
        tool_name = plan_json["tool"]
        args = plan_json["args"]
        # conf = plan_json["confidence"]  # (optional: log/trace)
        # thoughts = plan_json["thoughts"]

        if tool_name == "finish":
            final = args.get("final", last_answer or "No final answer provided.")
            critic = critic_grounding(final, [c.id for c in retrieved])
            if goal_met(final, critic, policy):
                return final + f"\n\n(Critic confidence: {critic.get('confidence', 0.0):.2f})"
            # If not good enough, continue planning

        elif tool_name in ("qa", "summarize", "skills", "compare_with_jd"):
            # Ensure we have context
            if not retrieved:
                retrieved = retrieve(collection, user_goal, n_results=4)

            if tool_name == "compare_with_jd":
                jd = args.get("jd")
                if not jd:
                    print("\n[Planner chose compare_with_jd] Paste the Job Description, then press Enter:")
                    jd = input().strip()
                result = TOOLS[tool_name](retrieved, jd)
            elif tool_name == "qa":
                q = args.get("question", user_goal)
                result = TOOLS[tool_name](retrieved, q)
            else:
                result = TOOLS[tool_name](retrieved)

            last_answer = result["output"]
            scratchpad.append({"step": step, "tool": tool_name, "obs": last_answer})

            # Self-check / critic
            critic = critic_grounding(last_answer, [c.id for c in retrieved])
            weak = bool(critic.get("unsupported")) or (critic.get("confidence", 0.0) < policy.min_critic_confidence)
            if weak:
                refinements = tool_refine_query(user_goal, last_answer)
                # Try best refinement to improve retrieval
                new_query = refinements[0]
                retrieved = retrieve(collection, new_query, n_results=4) or retrieved
                scratchpad.append({"step": step, "tool": "refine_query", "obs": f"retrieved={len(retrieved)} for '{new_query}'"})
                continue  # Re-plan

        elif tool_name == "write_email":
            subject = args.get("subject", "Exploring the Data Engineer role")
            body = args.get("body", "Hello,\n\nI'd love to connect regarding the role...\n")
            path = args.get("path", "./output_email.txt")
            result = tool_write_email_to_file(subject, body, path)
            last_answer = result["output"]
            scratchpad.append({"step": step, "tool": tool_name, "obs": last_answer})

        elif tool_name == "search_jobs":
            result = tool_search_jobs(args)
            last_answer = result["output"]
            scratchpad.append({"step": step, "tool": tool_name, "obs": last_answer})

        elif tool_name == "find_slots":
            result = tool_find_calendar_slots(args)
            last_answer = result["output"]
            scratchpad.append({"step": step, "tool": tool_name, "obs": last_answer})

        else:
            # Fallback
            result = tool_answer_question(retrieved, user_goal)
            last_answer = result["output"]
            scratchpad.append({"step": step, "tool": "qa", "obs": last_answer})

    # If we reached max steps without a clean finish:
    return last_answer + "\n\n(Note: Reached max planning steps without 'finish'.)"


# ======== MAIN ========
def main():
    print("Reading PDF...")
    pages = pdf_reader_with_pages(PDF_PATH)
    if not pages:
        print("No text found. Exiting.")
        return

    print("Splitting into chunks...")
    ids, docs, metas = textsplitter_with_meta(pages)

    print("Initializing vector DB...")
    collection = init_vector_db()
    upsert_documents(collection, ids, docs, metas, force_rebuild=True)
    print(f"Indexed {len(ids)} chunks into Chroma at {DB_PATH}.")

    while True:
        goal = input("\nEnter your goal (or 'exit'): ").strip()
        if goal.lower() == "exit":
            break
        answer = agent_loop(goal, collection, policy=GoalPolicy(
            require_citations=True,
            min_critic_confidence=0.6,
            max_steps=6
        ))
        print("\n--- Agentic Result ---")
        print(answer)


if __name__ == "__main__":
    main()
