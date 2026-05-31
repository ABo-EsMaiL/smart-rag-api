from fastapi import FastAPI, UploadFile, File
import os
import rag_engine

rag = rag_engine.RAGEngine()
app = FastAPI(title="ASK PDF")

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        with open(f"uploaded_{file.filename}", "wb") as f:
            f.write(contents)
        result = rag.ingest_pdf(f"uploaded_{file.filename}")
        if result:
            print("PDF ingested successfully.")
            return {"message": result}
        else:
            return {"message": "Failed to ingest PDF."}
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(f"uploaded_{file.filename}"):
            os.remove(f"uploaded_{file.filename}")
            
            
@app.post("/ask/")
async def ask_question(question: str):
    try:
        answer = rag.ask(question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
    