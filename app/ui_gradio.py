# Python 3.10
import os, requests, gradio as gr
API = os.getenv("API_URL", "http://127.0.0.1:8000/chat")

def fmt_sources(src_list):
    if not src_list: return ""
    lines = [f"{i+1}. {m.get('major','?').upper()} - {m.get('field','?')}" for i, m in enumerate(src_list)]
    return "Nguồn KB:\n" + "\n".join(lines)

def ask(message, history, top_k):
    try:
        r = requests.post(API, json={"query": message, "top_k": int(top_k)}, timeout=60)
        r.raise_for_status()
        data = r.json()
        ans = data.get("answer", "")
        extra = fmt_sources(data.get("sources", []))
        if extra: ans = f"{ans}\n\n---\n{extra}"
        back = data.get("backend","")
        if back: ans = f"{ans}\n\n_(backend: {back})_"
    except Exception as e:
        ans = f"Lỗi gọi API: {e}"

    history = history + [
        {"role":"user","content":message},
        {"role":"assistant","content":ans}
    ]
    return history, ""

with gr.Blocks(title="IU Chatbot (IT/CS/DS)") as demo:
    gr.Markdown("### 🤖 IU Chatbot — Giai đoạn 1 (IT/CS/DS)")
    with gr.Row():
        top_k = gr.Slider(1, 10, value=6, step=1, label="Top-K RAG")
    chat = gr.Chatbot(height=460, type="messages")
    box  = gr.Textbox(placeholder="Hỏi về IT/CS/DS (VI/EN)…", label="Your message")
    box.submit(ask, [box, chat, top_k], [chat, box])
    gr.Button("Send").click(ask, [box, chat, top_k], [chat, box])
    gr.Button("Clear").click(lambda: ([], ""), outputs=[chat, box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
