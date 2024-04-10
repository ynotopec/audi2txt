import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

#pipe
transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

import gradio as gr
import numpy as np

def transcribe(stream, new_chunk):
    sr, y = new_chunk
    y = y.astype(np.float32)
    
    # Convert stereo to mono if needed
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
#    state_value = stream if stream is not None else ""
    state_value = stream
    transcribed_text = transcriber({"sampling_rate": sr, "raw": stream})["text"]
#    transcribed_text_value = transcribed_text if transcribed_text else ""
    transcribed_text_value = transcribed_text
    return state_value, transcribed_text_value

demo = gr.Interface(
    fn=transcribe,
    inputs=["state", gr.Audio( streaming=True)],
    outputs=["state", "text"],
#gr.Textbox()],
    live=True,
)

demo.dependencies[0]["show_progress"] = False

demo.launch().queue()
