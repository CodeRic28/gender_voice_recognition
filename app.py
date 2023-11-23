from spectrogram import create_spectrogram
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import subprocess


model = load_model("model.h5")  # update with your model id
# pipe = pipeline("gender-voice-recognition", model=model)


# def transcribe_and_predict(spectrogram):
#     img = load_img(spectrogram, target_size=(150, 150))
#     x = img_to_array(img)
#     x = x / 255.0
#     x = np.expand_dims(x, axis=0)
#     # Make predictions using the model
#     predictions = model.predict(spectrogram)
#
#     if predictions[0][0] > predictions[0][1]:
#         return predictions[0][0]
#     return predictions[0][1]


def transcribe_and_predict(audio_waveform):
    # Assuming create_spectrogram returns a numpy array representing the spectrogram
    spectrogram = create_spectrogram(audio_waveform)

    # Resize the spectrogram to match the expected input size of the model
    resized_spectrogram = np.resize(spectrogram, (150, 150))

    # Normalize the spectrogram
    normalized_spectrogram = resized_spectrogram / np.max(np.abs(resized_spectrogram))

    # Expand dimensions to match the model's input shape
    x = np.expand_dims(normalized_spectrogram, axis=0)

    # Make predictions using the model
    predictions = model.predict(x)

    if predictions[0][0] > predictions[0][1]:
        return predictions[0][0]
    return predictions[0][1]


# Define the Gradio interface
demo = gr.Blocks()
mic_transcribe = gr.Interface(
    fn=transcribe_and_predict,
    inputs=gr.Audio(sources="upload", type="filepath", block_size=1024),
    outputs=gr.outputs.Textbox(),
)

# Launch Gradio interface
mic_transcribe.launch(inbrowser=True, inline=True,debug=True)



