import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# loading and visualizing audio files
import librosa
import librosa.display

# to play audio
import IPython.display as ipd

pwd = os.getcwd()
audio_mpath = os.path.join(pwd,f"dataset/validation/male")
audio_fpath = os.path.join(pwd,f"dataset/validation/female")

audio_mclips = os.listdir(audio_mpath)
audio_fclips = os.listdir(audio_fpath)

output_directory = "./SAVING"
os.makedirs(output_directory, exist_ok=True)

def create_spectrogram(input_mp3_file, output_directory):
    # Extract the filename without extension
    file_name = os.path.splitext(os.path.basename(input_mp3_file))[0]
    print(file_name)
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

image_path = os.path.join('cv-other-test','sample-002951.mp3')
create_spectrogram(image_path,output_directory)

# for file in audio_fclips:
#     print(f"Creating spectrogram for {file}")
#     audio_file_path = os.path.join(audio_fpath,file)
#     audio_signal, sample_rate = librosa.load(audio_file_path, sr=None)
#     plt.figure(figsize=(14, 5))
#
#     # Convert audio waveform to spectrogram
#     X = librosa.stft(audio_signal)  # fourier transform
#     Xdb = librosa.amplitude_to_db(abs(X))
#
#     # display the spectrogram
#     # librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz')
#     librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='log')
#     plt.colorbar()
#
#     # Save the spectrogram as a PNG file
#     spectrogram_filename = os.path.splitext(file)[0]
#     spectrogram_filepath = os.path.join(output_directory, spectrogram_filename)
#     plt.savefig(spectrogram_filepath, bbox_inches='tight')
#
#     # Close the current figure to release memory
#     plt.close()



# print(len(audio_fclips),len(audio_mclips))
#
# audio_signal, sample_rate = librosa.load(audio_fpath+'/'+audio_fclips[20],sr=None)
#
# ipd.Audio(audio_signal, rate=sample_rate)
#
# plt.figure(figsize=(14, 5))
# librosa.display.waveshow(audio_signal, sr=sample_rate)
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.title("Waveplot of Audio")
#
# # Show the waveplot
# # plt.show()
#
# # Convert audio waveform to spectrogram
# X = librosa.stft(audio_signal) # fourier transform
# Xdb = librosa.amplitude_to_db(abs(X))
# plt.figure(figsize=(14, 5))
# librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz')
# plt.colorbar()
# # plt.show()
#
# # Applying log transformation on the loaded audio signals
# plt.figure(figsize=(14, 5))
# librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='log')
# plt.colorbar()
# plt.show()