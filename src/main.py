"""
Local AI Assistant - Ana Uygulama.

FastAPI ana sunucu olarak çalışır.
Gradio UI, FastAPI'ye /chat altında mount edilir.
API endpoint'leri /api altında.

Çalıştırma:
    python -m src.main
"""

import gradio as gr
import uvicorn
from fastapi import FastAPI

from src.chat.router import router as chat_router
from src.config import get_settings
from src.ui.app import create_ui


def create_app() -> FastAPI:
    """FastAPI uygulamasını oluşturur."""
    settings = get_settings()

    app = FastAPI(
        title=settings.ui.title,
        version="0.2.0",
    )

    app.include_router(chat_router, prefix="/api")

    ui = create_ui()
    app = gr.mount_gradio_app(app, ui, path="/chat")

    return app


def main() -> None:
    """Uygulamayı başlatır."""
    settings = get_settings()
    app = create_app()

    print(f"\n  Local AI Assistant v0.2.0")
    print(f"  UI:   http://localhost:{settings.server.port}/chat")
    print(f"  API:  http://localhost:{settings.server.port}/api")
    print(f"  Docs: http://localhost:{settings.server.port}/docs")
    print(f"  Model: {settings.llm.model}\n")

    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
    )


if __name__ == "__main__":
    main()
