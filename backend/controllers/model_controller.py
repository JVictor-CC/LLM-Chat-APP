from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from huggingface_hub import login, logout


class ModelController:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.logged_in = False
        self.model_name = None

    def _login(self, hf_token):
        if hf_token:
            login(hf_token)
            self.logged_in = True

    def _logout(self):
        if self.logged_in:
            logout()
            self.logged_in = False

    def load_model(self, model_name: str, hf_token: str | None = None):
        try:
            if self.model is None:
                self._login(hf_token)

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)

                self._logout()

                response = {
                    "data": {
                        "type": "Model Load",
                        "message": "Model loaded successfully",
                    }
                }
                status_code = 200
                self.model_name = model_name
            else:
                response = {
                    "data": {
                        "type": "Model Load",
                        "message": "Model already loaded",
                    }
                }
                status_code = 200
        except Exception as ex:
            response = {
                "data": {
                    "type": "Error",
                    "message": ex
                }
            }
            status_code = 500

        return status_code, response

    def load_peft_model(self, model_name: str, base_model: str, hf_token: str | None = None):
        try:
            if self.model is None:
                self._login(hf_token)

                config = PeftConfig.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                self.model = AutoModelForCausalLM.from_pretrained(base_model)
                self.model = PeftModel.from_pretrained(self.model, model_name)

                self._logout()

                response = {
                    "data": {
                        "type": "Peft Model Load",
                        "message": "Peft Model loaded successfully",
                    }
                }
                status_code = 200

                self.model_name = model_name
            else:
                response = {
                    "data": {
                        "type": "Peft Model Load",
                        "message": "Peft Model already loaded",
                    }
                }
                status_code = 200

        except Exception as ex:
            response = {
                "data": {
                    "type": "Error",
                    "message": ex
                }
            }
            status_code = 500

        return status_code, response

    def generate(self, prompt: str, new_tokens: int | None = None) -> Dict:

        if self.tokenizer is None or self.model is None:
            return {
                "data": {
                    "type": "Error",
                    "message": "Model not loaded"
                }
            }

        if new_tokens is None:
            new_tokens = 512

        messages = [
            {'role': 'user', 'content': prompt}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt")
        outputs = self.model.generate(inputs, max_new_tokens=new_tokens, do_sample=False, top_k=50,
                                      top_p=0.95, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(
            outputs[0][len(inputs[0]):], skip_special_tokens=True)

        return {
            "data": {
                "type": "Model Inference",
                "prompt": prompt,
                "response": response,
            }
        }

    def save_model(self):
        if self.model:
            folder_name = self.model_name.split("/")[-1]
            path = f"local_model/{folder_name}"
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

            return {
                "data": {
                    "type": "Model Save",
                    "message": "Model saved successfully"
                }
            }
        else:
            return {
                "data": {
                    "type": "Error",
                    "message": "Model is not loaded"
                }
            }

    def unload_model(self):
        if self.model is not None:
            del self.model
            del self.tokenizer

            self.model = None
            self.tokenizer = None
            self.model_name = None

            return {
                "data": {
                    "type": "Model Unload",
                    "message": "Model unloaded successfully"
                }
            }
        else:
            return {
                "data": {
                    "type": "Error",
                    "message": "There is no Model loaded"
                }
            }

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model
