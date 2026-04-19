import os
import json
import re
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
JOURNAL_FILE  = "./growth_journal.json"
DATA_PATH     = os.path.join(os.path.dirname(__file__), "data", "Green Assistant Dataset.docx")

rag_chain = None

# ── Journal helpers ───────────────────────────────────────────────────────────
def load_journal():
    if os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_journal(entries):
    with open(JOURNAL_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

# ── Startup ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    print("BioSprout AI - Loading...")

    loader  = Docx2txtLoader(DATA_PATH)
    docs    = loader.load()
    splits  = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)

    # Free local embeddings — no Ollama needed
    embeddings   = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore  = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db_v2")

    # Groq LLM — fast & free
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=512,
    )

    system_prompt = (
        "You are an expert Arabic agricultural assistant called 'مساعدي الأخضر'. "
        "Answer ONLY based on the context below. "
        "Always reply in Arabic. Be concise and practical.\n\nContext: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    retriever  = vectorstore.as_retriever(search_kwargs={"k": 4})
    qa_chain   = create_stuff_documents_chain(llm, prompt)
    rag_chain  = create_retrieval_chain(retriever, qa_chain)

    print("BioSprout AI - Ready on http://localhost:8000")
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="BioSprout AI API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Models ────────────────────────────────────────────────────────────────────
class SeedAnalysisRequest(BaseModel):
    plant_name: str
    description: str = ""

class ChatRequest(BaseModel):
    message: str

class GrowthEntry(BaseModel):
    plant_name: str
    stage: str
    notes: str
    date: str = ""

# ── Calendars & Care data ─────────────────────────────────────────────────────
CALENDARS = {
    "التفاح":    {"best_months": [1,2,3,11,12], "good_months": [4,10],   "avoid_months": [6,7,8],  "best_season": "الشتاء والخريف",   "harvest_season": "يوليو – أكتوبر"},
    "المانجو":   {"best_months": [3,4,5],        "good_months": [2,6],    "avoid_months": [11,12,1],"best_season": "الربيع",            "harvest_season": "يونيو – أغسطس"},
    "الفراولة":  {"best_months": [9,10,11],      "good_months": [8,12],   "avoid_months": [6,7],    "best_season": "الخريف",            "harvest_season": "ديسمبر – أبريل"},
}

CARE_GUIDES = {
    "التفاح": {
        "irrigation":  {"amount_liters": "5–10 لتر",   "frequency": "مرتان أسبوعياً",          "method": "ري بالتقطير أو الغمر"},
        "fertilizer":  {"type": "NPK 10-10-10",         "schedule": "كل 6 أسابيع",               "organic": "سماد عضوي شهرياً"},
        "planting":    {"depth": "30–40 سم",            "spacing": "4–6 متر",                    "soil": "تربة طينية رملية pH 6–7"},
        "temperature": {"ideal": "15–24°C",             "min": "4°C",                            "max": "38°C"},
    },
    "المانجو": {
        "irrigation":  {"amount_liters": "15–20 لتر",  "frequency": "3 مرات أسبوعياً صيفاً",   "method": "ري عميق حول الجذع"},
        "fertilizer":  {"type": "بوتاسيوم وفوسفور",    "schedule": "3 مرات سنوياً",             "organic": "سماد بلدي متحلل"},
        "planting":    {"depth": "50–60 سم",            "spacing": "8–10 متر",                   "soil": "تربة طميية pH 5.5–7.5"},
        "temperature": {"ideal": "24–30°C",             "min": "10°C",                           "max": "45°C"},
    },
    "الفراولة": {
        "irrigation":  {"amount_liters": "1–2 لتر",    "frequency": "يومياً أو كل يومين",       "method": "ري بالتنقيط للجذور"},
        "fertilizer":  {"type": "نيتروجين للأوراق",    "schedule": "كل 3 أسابيع",               "organic": "سماد الدبال"},
        "planting":    {"depth": "15–20 سم",            "spacing": "30 سم",                      "soil": "تربة خفيفة pH 5.5–6.5"},
        "temperature": {"ideal": "15–22°C",             "min": "0°C",                            "max": "30°C"},
    },
}

def match_plant(name: str, data: dict):
    for key, val in data.items():
        if key in name or name in key:
            return {**val, "plant_name": name}
    return None

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/analyze-seed")
async def analyze_seed(req: SeedAnalysisRequest):
    query = (
        f"هل بذور {req.plant_name} صالحة للزراعة؟ "
        "أجب بـ JSON فقط بهذا الشكل بدون أي نص آخر: "
        '{"viability_percent": <0-100>, "success_rate": <0-100>, '
        '"status": "<صالحة أو غير صالحة>", "recommendation": "<جملة>", '
        '"best_soil": "<نوع التربة>", "notes": "<ملاحظة>"}'
    )
    result = rag_chain.invoke({"input": query})
    answer = result.get("answer", "")
    try:
        m = re.search(r"\{.*\}", answer, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {"viability_percent": 75, "success_rate": 70,
            "status": "صالحة للزراعة",
            "recommendation": f"بذور {req.plant_name} تبدو صالحة في الظروف المناسبة",
            "best_soil": "تربة خصبة جيدة التصريف",
            "notes": "احفظ البذور في مكان جاف وبارد"}

@app.get("/planting-calendar/{plant_name}")
async def get_planting_calendar(plant_name: str):
    result = rag_chain.invoke({"input": f"ما أفضل أشهر زراعة {plant_name}؟"})
    notes  = result.get("answer", "")[:400]
    cal    = match_plant(plant_name, CALENDARS)
    if cal:
        return {**cal, "notes": notes}
    return {"plant_name": plant_name, "best_months": [3,4,10,11],
            "good_months": [2,5,9,12], "avoid_months": [7,8],
            "best_season": "الربيع والخريف", "harvest_season": "حسب النوع", "notes": notes}

@app.get("/care-guide/{plant_name}")
async def get_care_guide(plant_name: str):
    result = rag_chain.invoke({"input": f"اذكر كمية الري والسماد وطريقة الغرس لـ{plant_name}"})
    tips   = result.get("answer", "")[:500]
    guide  = match_plant(plant_name, CARE_GUIDES)
    if guide:
        return {**guide, "tips": tips}
    return {"plant_name": plant_name,
            "irrigation":  {"amount_liters": "حسب الحجم", "frequency": "حسب الطقس", "method": "ري منتظم"},
            "fertilizer":  {"type": "سماد متوازن", "schedule": "شهرياً", "organic": "سماد عضوي"},
            "planting":    {"depth": "حسب البذرة", "spacing": "حسب النوع", "soil": "تربة خصبة"},
            "temperature": {"ideal": "20–28°C", "min": "10°C", "max": "40°C"},
            "tips": tips}

@app.get("/growth-journal")
async def get_growth_journal():
    entries = load_journal()
    return {"entries": entries, "total": len(entries)}

@app.post("/growth-journal/add")
async def add_growth_entry(entry: GrowthEntry):
    entries   = load_journal()
    new_entry = {"id": max((e.get("id",0) for e in entries), default=0) + 1,
                 "plant_name": entry.plant_name, "stage": entry.stage,
                 "notes": entry.notes,
                 "date": entry.date or datetime.now().strftime("%Y-%m-%d"),
                 "created_at": datetime.now().isoformat()}
    entries.append(new_entry)
    save_journal(entries)
    return {"success": True, "entry": new_entry}

@app.delete("/growth-journal/{entry_id}")
async def delete_growth_entry(entry_id: int):
    entries = [e for e in load_journal() if e.get("id") != entry_id]
    save_journal(entries)
    return {"success": True}

@app.post("/chat")
async def chat(req: ChatRequest):
    result = rag_chain.invoke({"input": req.message})
    return {"answer": result.get("answer", "عذراً، لم أتمكن من معالجة طلبك")}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
