import requests
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction


class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name = 'nomic-embed-text'):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/embeddings"
    
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = requests.post(self.url, json={"model": self.model_name, "prompt": text})
            if response.status_code == 200:
                embedding = response.json().get("embedding")
                embeddings.append(embedding)
            else:
                raise Exception(f"Failed to get embedding for text: {text}. Status code: {response.status_code}")
        return embeddings

class RAGEngine:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="company_docs", embedding_function=OllamaEmbeddingFunction())
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.llm_url = "http://localhost:11434/api/generate"
        
    def ingest_pdf(self, pdf_path: str):
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        print(f"✅ تم استخراج النص، طوله: {len(full_text)} حرف")

        if not full_text.strip():
            return "الملف فاضي أو عبارة عن صور (Scanned) ومفيش نص ممكن قراءته!"
        
        chunks = self.text_splitter.split_text(full_text)
        print(f"✅ تم تقطيع النص إلى: {len(chunks)} قطعة (Chunks)")
        
        if not chunks:
            return "مفيش قطع (Chunks) اتستخرج من النص!"
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        print(f"✅ تم إنشاء: {len(ids)} باركود (IDs)")

        print(f"✅ تم استخراج النص، طوله: {len(full_text)} حرف")
        print(f"✅ تم تقطيع النص إلى: {len(chunks)} قطعة (Chunks)")
        print(f"✅ تم إنشاء: {len(ids)} باركود (IDs)")

        self.collection.add(ids=ids, documents=chunks)
        return f"تم بنجاح تخزين {len(chunks)} قطعة في قاعدة البيانات!"
    
    def ask(self, question: str):
        results = self.collection.query(
            query_texts=[question],
            n_results=5
        )
        
        context = "\n".join(results['documents'][0] if results['documents'] else "no relevant context found")
        
         # 🔍 كاميرا المراقبة: طباعة السياق في الـ Terminal عشان نشوفه
        print("\n" + "="*50)
        print("🔍 السياق اللي الـ ChromaDB رجعه (Context):")
        print(context)
        print("="*50 + "\n")
        
        prompt = f"""
                You are a professional AI assistant.
                Your task is to answer the user's question based ONLY on the provided context.
                If the answer is not in the context, you MUST say exactly: "المعلومة دي مش متاحة في السياق."

                CRITICAL RULES (قواعد صارمة):
                1. You MUST answer in SIMPLE ARABIC (اللغة العربية الفصحى المبسطة).
                2. DO NOT use any English words in your answer.
                3. DO NOT use any other languages (No Russian, No French, etc.).
                4. DO NOT mix languages. Translate any English names in the context to Arabic if possible, or keep them as proper nouns only.

                --- Context ---
                {context}
                --- End of Context ---

                User Question: {question}

                Simple Arabic Answer:
                """

        payload = {"model": "qwen2.5", "prompt": prompt, "stream": False}
        response = requests.post(self.llm_url, json=payload)
        return response.json().get("response", "حصل خطأ في الموديل")