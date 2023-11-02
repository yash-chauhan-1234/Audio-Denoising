import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import librosa


class CustomDataGenerator(Sequence):
    
    def __init__(self, audio_signal, sample_rate, target_size=None):
    
        self.audio_signal=audio_signal
        self.target_size=target_size
        self.sr=sample_rate
        self.target_avg_length=3
#         self.scaler = MinMaxNormaliser(0,1)
        
    # def on_epoch_end(self):
    #     if self.shuffle:
    #         self.df = self.df.sample(frac=1).reset_index(drop=True)
            
    # def __len__(self):
    #     return self.n // self.batch_size
    
    # def __getitem__(self,index):
    
    #     batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size,:]
    #     X1, X2 = self.__get_data(batch)        
    #     return X1, X2
    
    # def __scale(self, array, rescale=1./255):
    #     array=array*rescale
    #     return array
    
    # def __padding(self, spec):
    #     a,b = spec.shape
    #     h = np.zeros((1032-a,b))
    #     v = np.zeros((1032,632-b))
    #     result = np.vstack([spec,h])
    #     result = np.hstack([result,v])
    #     return result
    
    def __get_spectogram(self, audio, sr):
        target_samples = int(self.target_avg_length * sr)
        current_samples = len(audio)

        if current_samples < target_samples:
            # Pad the audio with zeros
            padding = target_samples - current_samples
            audio = np.pad(audio, (0, padding), mode='constant')
        elif current_samples > target_samples:
            # Truncate the audio
            audio = audio[:target_samples]
        # Compute the mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256)
        
        # Plot the mel spectrogram
        qm=librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max))
        return qm.to_rgba(qm.get_array())[:,:,:3]
    
#     def __extract_spectrograms(self):
# #         x , sr = librosa.load(audio,sr=16000)
# #         X = librosa.stft(x)
# #         Xdb = librosa.amplitude_to_db(np.abs(X),)
# #         return Xdb
#         image = tf.io.read_file(dest)
#         image = tf.image.decode_png(image, channels=3)
#         image=image[55:430, 90:575, :]
#         if self.target_size is not None:
#             image=tf.image.resize(image, self.target_size)
#         return image
    
#     def __get_data(self,batch):
        
#         X1, X2 = list(), list()
#         src_audios = batch[self.X_col].tolist()
#         target_audios = batch[self.y_col].tolist()
        
#         for src,target in zip(src_audios,target_audios):
#             input_spec = self.__extract_spectrograms(src)
#             output_spec = self.__extract_spectrograms(target)
# #             output_spec = tf.reshape(output_spec, self.reshape)
#             X1.append(input_spec)
#             X2.append(output_spec)
            
#         X1, X2 = np.array(X1), np.array(X2)
#         X1 = self.__scale(X1)
#         X2 = self.__scale(X2)
# #         X2 = self.scaler.normalise(X2)        
#         return X1, X2
    
    def __get_one_data(self):
        image=self.__get_spectogram(self.audio_signal, self.sr)
        print(image.shape)
        # image=image[55:430, 90:575, :]
        if self.target_size is not None:
            image=tf.image.resize(image, self.target_size)
        # image*=1./255
        return image
    
    def get_image(self):
        return self.__get_one_data()
