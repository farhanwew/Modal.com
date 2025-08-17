from modal import App
from .app_modal import app as llm_app

app = App("eros-deploy")
app.include(llm_app)

if __name__ == "__main__":
    # modal deploy -m src.deploy
    pass