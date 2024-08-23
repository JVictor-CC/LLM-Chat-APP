from fastapi import FastAPI
from routers.model_router import model_router

app = FastAPI()

app.include_router(model_router)

@app.get("/")
def root():
    return {"message": "Welcome to the API!"}
