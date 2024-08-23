from fastapi import APIRouter, HTTPException
from controllers.model_controller import ModelController
from schemas import ModelLoadSchema, PeftModelLoadSchema, InferenceSchema

model_router = APIRouter()
model_controller = ModelController()


@model_router.post("/load_model")
async def load_model(input_model: ModelLoadSchema):
    result = model_controller.load_model(
        input_model.model_name, input_model.hf_token)
    return result


@model_router.post("/load_peft_model")
async def load_peft_model(input_model: PeftModelLoadSchema):
    result = model_controller.load_peft_model(
        input_model.model_name, input_model.base_model, input_model.hf_token)
    return result


@model_router.post("/save_model")
async def save_model():
    result = model_controller.save_model()
    return result


@model_router.delete("/unload_model")
async def unload_model():
    result = model_controller.unload_model()
    return result


@model_router.post("/inference")
async def predict(input_text: InferenceSchema):
    if model_controller.model is None:
        raise HTTPException(
            status_code=500, detail="Nenhum modelo carregado. Por favor, carregue um modelo antes de fazer inferências.")
    result = model_controller.generate(
        input_text.prompt, input_text.new_tokens)
    return result
