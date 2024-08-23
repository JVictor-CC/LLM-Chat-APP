import requests
import streamlit as st

if 'is_loaded' not in st.session_state:
    st.session_state.is_loaded = False
    st.session_state.basemodel_name = None
    st.session_state.model_name = None
    
@st.fragment
def sidebar():
    if not st.session_state.is_loaded:

        peft = st.checkbox("Use Peft", False)

        st.session_state.model_name = st.text_input(
            "Model Name *", placeholder="Enter the model name")

        if peft:
            st.session_state.basemodel_name = st.text_input(
                "Base Model Name", placeholder="Enter the base model name")

        hf_token = st.text_input(
            "HuggingFace Token", placeholder="Enter your HF token", type="password")

        if st.button("Load Model"):
            if st.session_state.model_name:
                if peft and not st.session_state.basemodel_name:
                    st.error("Please enter the base model name!")
                else:
                    try:
                        payload = {
                            "model_name": st.session_state.model_name,
                            "hf_token": hf_token
                        }
                        if peft and st.session_state.basemodel_name:
                            payload["basemodel_name"] = st.session_state.basemodel_name

                        url = "http://tcc-backend-container:8000/load_peft_model" if peft else "http://tcc-backend-container:8000/load_model"

                        with st.spinner('Loading model...'):
                            response = requests.post(url, json=payload)
                            
                            if response.status_code == 200:
                                if peft:
                                    st.success(
                                        "Peft model loaded successfully!")
                                else:
                                    st.success("Model loaded successfully!")
                                st.session_state.is_loaded = True
                                st.rerun(scope="fragment")
                            else:
                                st.error(
                                    f"Loading model failed with status code: {response.status_code}")
                                st.write(response.text)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.error("Please enter the model name!")
    else:
        st.write(f'Model: {st.session_state.model_name}')
        if st.button("Unload Model"):
            try:
                url = "http://tcc-backend-container:8000/unload_model"

                with st.spinner('Unloading model...'):
                    response = requests.delete(url)
                    response = response.json()

                    if response["data"]["type"] != "Error":
                        st.toast(response["data"]["message"], icon="✅")
                        st.session_state.is_loaded = False
                        st.session_state.basemodel_name = None
                        st.session_state.model_name = None
                        st.rerun(scope="fragment")
                    else:
                        st.error(
                            response["data"]["message"])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                
        if st.button("Save Model"):
            try:
                url = "http://tcc-backend-container:8000/save_model"

                with st.spinner('Saving Model...'):
                    response = requests.post(url, {})
                    response = response.json()

                    if response["data"]["type"] != "Error":
                        st.toast(response["data"]["message"], icon="✅")
                    else:
                        st.error(
                            response["data"]["message"])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")    


st.header('Chat', divider='rainbow')

# Sidebar
with st.sidebar:
    st.title("StudyLLM")
    sidebar()

# Chat Messages

new_tokens = st.text_input("Max tokens to be generated", value="512")
messages = st.container()

# Chat Input

prompt = st.chat_input("Write your message here...")
if prompt:
    try:
        payload = {
            "prompt": prompt,
            "new_tokens": int(new_tokens),
        }
        messages.chat_message("user").write(prompt)
        
        with st.spinner('Performing inference...'):
            response = requests.post(
                "http://tcc-backend-container:8000/inference", json=payload)

            if response.status_code == 200:
                result = response.json()
                result = result["data"]["response"]
                messages.chat_message("assistant").write(f"Machine: {result}")
                st.toast("Inference successful!", icon="✅")
            else:
                result = response.json()
                st.error(result["detail"])

    except Exception as e:
        st.toast(":red-background[API error!]", icon="❌")
        st.error(f"An error occurred: {str(e)}")
else:
    st.error("Please enter a message!")
