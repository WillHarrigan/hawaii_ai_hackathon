from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from llm import process_query

app = FastAPI()

# Set up the Jinja2 templates directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="template")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("Kani 15.html", {"request": request})


@app.post("/message")
async def on_message(request: Request):
    data = await request.json()
    # Retrieve the "message" value from the JSON
    user_message = data.get("message")
    response = await process_query(user_message)
    print("Kani:", response)
    return {"message" : response}


@app.get("/plot", response_class=FileResponse)
def render_plot():
    # Ensure the path is correct relative to where the app runs
    return FileResponse("output_files/ndvis.html", media_type="text/html")