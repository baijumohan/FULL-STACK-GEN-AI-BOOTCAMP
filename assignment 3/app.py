import base64
import io
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS
from huggingface_hub import InferenceClient
from PIL import Image

try:
    import google.generativeai as genai
except Exception:
    genai = None


load_dotenv(Path(__file__).with_name(".env"))


PROVIDERS = {
    "Google": {
        "models": ["gemini-2.0-flash", "gemini-2.5-flash"],
        "default": "gemini-2.0-flash",
    },
    "Groq": {
        "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        "default": "llama-3.1-8b-instant",
    },
    "OpenRouter": {
        "models": ["openai/gpt-4o-mini", "meta-llama/llama-3.1-8b-instruct"],
        "default": "meta-llama/llama-3.1-8b-instruct",
    },
    "Hugging Face": {
        "models": ["meta-llama/Llama-3.2-1B-Instruct"],
        "default": "meta-llama/Llama-3.2-1B-Instruct",
    },
}


def _safe_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("content") or value.get("text") or str(value)
    if hasattr(value, "content"):
        return str(value.content)
    return str(value)


def _to_base64(file_bytes):
    return base64.b64encode(file_bytes).decode("utf-8")


def _google_model_name(model_name):
    return model_name if model_name.startswith("models/") else f"models/{model_name}"


def generate_text_with_google(prompt, model_name):
    if genai is None:
        raise RuntimeError("google-generativeai is not installed")
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it to .env or Streamlit secrets.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(_google_model_name(model_name))
    response = model.generate_content(prompt)
    return _safe_text(response.text)


def generate_text_with_groq(prompt, model_name):
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    resp = requests.post(url, headers=headers, json=data, timeout=120)
    resp.raise_for_status()
    payload = resp.json()
    return _safe_text(payload["choices"][0]["message"]["content"])


def generate_text_with_openrouter(prompt, model_name):
    api_key = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Multimodal Free Models App",
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(url, headers=headers, json=data, timeout=120)
    resp.raise_for_status()
    payload = resp.json()
    return _safe_text(payload["choices"][0]["message"]["content"])


def generate_text_with_hf(prompt, model_name):
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is missing. Add it to .env or Streamlit secrets.")
    client = InferenceClient(model=model_name, token=token)
    try:
        response = client.chat_completion([{"role": "user", "content": prompt}], max_tokens=180)
        return _safe_text(response.choices[0].message.content)
    except Exception:
        response = client.text_generation(prompt, max_new_tokens=180)
        return _safe_text(response)


def generate_text(provider, model_name, prompt):
    if provider == "Google":
        return generate_text_with_google(prompt, model_name)
    if provider == "Groq":
        return generate_text_with_groq(prompt, model_name)
    if provider == "OpenRouter":
        return generate_text_with_openrouter(prompt, model_name)
    if provider == "Hugging Face":
        return generate_text_with_hf(prompt, model_name)
    raise ValueError("Unsupported provider")


def describe_image(provider, model_name, uploaded_file, prompt_text):
    if uploaded_file is None:
        return "Please upload an image first."

    image = Image.open(uploaded_file).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    if provider == "Google":
        if genai is None:
            raise RuntimeError("google-generativeai is not installed")
        api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_google_model_name(model_name or "gemini-2.0-flash"))
        response = model.generate_content([prompt_text or "Describe this image in detail.", image])
        return _safe_text(response.text)

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
    client = InferenceClient(model="nlpconnect/vit-gpt2-image-captioning", token=token)
    try:
        caption = client.image_to_text(image_bytes)
    except Exception:
        caption = client.image_to_text(image)
    return _safe_text(caption)


def describe_video(provider, model_name, uploaded_file, prompt_text):
    if uploaded_file is None:
        return "Please upload a video first."

    if provider == "Google":
        if genai is None:
            raise RuntimeError("google-generativeai is not installed")
        api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_google_model_name(model_name or "gemini-2.0-flash"))
        response = model.generate_content([prompt_text or "Explain what is happening in this video.", uploaded_file])
        return _safe_text(response.text)

    # Best-effort fallback: inspect the first frame only for free providers.
    try:
        video = Image.open(uploaded_file)
        video = video.convert("RGB")
        buf = io.BytesIO()
        video.save(buf, format="PNG")
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
        client = InferenceClient(model="nlpconnect/vit-gpt2-image-captioning", token=token)
        caption = client.image_to_text(buf.getvalue())
        return f"First-frame caption: {_safe_text(caption)}"
    except Exception as exc:
        return f"Video analysis is limited in this free setup. Please use Google as the provider for direct video input. Error: {exc}"


def generate_image(prompt):
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
    client = InferenceClient(model="black-forest-labs/FLUX.1-schnell", token=token)
    image = client.text_to_image(prompt, model="black-forest-labs/FLUX.1-schnell")
    return image


def text_to_audio(text):
    buf = io.BytesIO()
    gTTS(text=text, lang="en").write_to_fp(buf)
    buf.seek(0)
    return buf


def main():
    st.set_page_config(page_title="Free Multimodal Studio", layout="wide")
    st.title("Free Multimodal Studio")
    st.caption("Text, image, and video input with text/image/audio output using free providers: Google, Groq, OpenRouter, and Hugging Face.")

    with st.sidebar:
        st.header("Provider")
        provider = st.selectbox("Choose backend", list(PROVIDERS.keys()), index=0)
        model_name = st.selectbox("Model", PROVIDERS[provider]["models"], index=0)
        st.caption("Tip: add your keys to the project .env file before running.")

    input_modality = st.radio("Input modality", ["Text", "Image", "Video"], horizontal=True)

    prompt_text = st.text_area("Prompt / instruction", value="Describe the input and answer in a helpful way.")

    uploaded_file = None
    if input_modality == "Image":
        uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    elif input_modality == "Video":
        uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

    output_modality = st.selectbox("Output modality", ["Text", "Image", "Audio"])

    if st.button("Generate", type="primary"):
        try:
            if input_modality == "Text":
                answer = generate_text(provider, model_name, prompt_text)
            elif input_modality == "Image":
                answer = describe_image(provider, model_name, uploaded_file, prompt_text)
            else:
                answer = describe_video(provider, model_name, uploaded_file, prompt_text)

            if output_modality == "Text":
                st.subheader("Text output")
                st.write(answer)

            elif output_modality == "Image":
                st.subheader("Image output")
                if input_modality == "Text":
                    image = generate_image(prompt_text)
                else:
                    image = generate_image(f"Create a visual summary of: {answer}")
                st.image(image, caption="Generated image from a free Hugging Face model")

            else:
                st.subheader("Audio output")
                audio_buf = text_to_audio(answer if answer else prompt_text)
                st.audio(audio_buf.getvalue(), format="audio/mp3")

        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            st.info("Make sure your free API keys are present in the project .env file before running the app.")


if __name__ == "__main__":
    main()
