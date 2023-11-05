import os
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import gradio as gr

# Load the pre-trained model
model = tf.keras.models.load_model("model.h5")

path = os.path.join(os.getcwd(),'cv-valid-test/sample-000006.mp3')
def create_spectrogram(input_mp3_file, output_directory):
    # Extract the filename without extension
    file_name = os.path.splitext(os.path.basename(input_mp3_file))[0]

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load the audio signal from the MP3 file
    audio_signal, sample_rate = librosa.load(input_mp3_file, sr=None)

    # Create a figure to plot the spectrogram
    plt.figure(figsize=(14, 5))

    # Convert audio waveform to spectrogram
    X = librosa.stft(audio_signal)  # Fourier transform
    Xdb = librosa.amplitude_to_db(abs(X))

    # Display the spectrogram
    librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar()
    # Save the spectrogram as a PNG file
    spectrogram_filepath = os.path.join(output_directory, f"{file_name}_spectrogram.png")
    plt.savefig(spectrogram_filepath, bbox_inches='tight')

    # Close the current figure to release memory
    plt.close()

    return spectrogram_filepath

def classify_spectrogram(spectrogram_path):
    # Load the spectrogram as an image
    img = tf.keras.preprocessing.image.load_img(spectrogram_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Predict the spectrogram using the loaded model
    predictions = model.predict(img_array)
    if predictions[0] > 0.5:
        return "Female"
    else:
        return "Male"

# # Define Gradio interface
input_mp3 = gr.inputs.File(label="Input MP3 Audio")
output_spectrogram = gr.outputs.Image(label="Spectrogram Image")
output_gender = gr.outputs.Text(label="Gender Prediction")


# Create Gradio app
gr.Interface(
    fn=[create_spectrogram, classify_spectrogram],
    inputs=input_mp3,
    outputs=[output_spectrogram, output_gender],
    live=True
).launch()