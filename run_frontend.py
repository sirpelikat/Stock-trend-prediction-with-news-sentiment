import uvicorn
import webbrowser
import time
import threading

def open_browser():
    time.sleep(1.5)
    print("Opening browser at http://127.0.0.1:8000 ...")
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    print("=" * 60)
    print(" Stock Trend & Sentiment Dashboard (CPC353 NLP Project)")
    print(" Starting local FastAPI server on http://127.0.0.1:8000")
    print("=" * 60)
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

