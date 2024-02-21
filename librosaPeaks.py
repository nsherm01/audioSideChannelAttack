import matplotlib.pyplot as plt
import librosa, librosa.display
import numpy

y, sr = librosa.load('audiofiles/file_example_WAV_2MG.wav')
'''plt.figure()
plt.subplot(3, 1, 2)
librosa.display.waveshow(y, sr=sr)
plt.title('Stereo')

plt.show()'''

hop_length = 256
onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

N = len(y)
T = N/float(sr)
t = numpy.linspace(0, T, len(onset_envelope))

plt.figure(figsize=(14, 5))
plt.plot(t, onset_envelope)
plt.xlabel('Time (sec)')
plt.xlim(xmin=0)
plt.ylim(0)
#plt.show()

onset_frames = librosa.util.peak_pick(onset_envelope, pre_max=7, post_max=7, pre_avg=7, post_avg=7, delta=0.5, wait=5)

'''
plt.figure(figsize=(14, 5))
plt.plot(t, onset_envelope)
plt.grid(False)
plt.vlines(t[onset_frames], 0, onset_envelope.max(), color='r', alpha=0.7)
plt.xlabel('Time (sec)')
plt.xlim(0, T)
plt.ylim(0)
#plt.show()
'''

onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)

# Add end sample index
onset_samples = numpy.append(onset_samples, len(y))

# Save each note as a separate .wav file
for i in range(len(onset_samples) - 1):
    start_sample = onset_samples[i]
    end_sample = onset_samples[i + 1]
    note = y[start_sample:end_sample]
    librosa.output.write_wav(f'note_{i}.wav', note, sr)
