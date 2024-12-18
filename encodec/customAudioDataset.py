import os
import torch
from torch.utils.data import Dataset
import librosa


class CustomAudioDataset(Dataset):
    def __init__(self, dataset_folder, n_samples=400, transform=None, sample_rate=24000, channels=1, tensor_cut=15000):
        self.audio_folder = dataset_folder
        self.transform = transform
        self.sample_rate = sample_rate
        self.channels = channels
        self.tensor_cut = tensor_cut
        self.audio_files = load_audio_files(dataset_folder)[:n_samples]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform, sample_rate = librosa.load(audio_path, sr=self.sample_rate, mono=self.channels==1)        
        
        waveform = torch.as_tensor(waveform)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
            waveform = waveform.expand(self.channels, -1)
        
        # Apply any additional transformations
        if self.transform:
            waveform = self.transform(waveform)
            
        #  trim the tensor to a fixed length
        if self.tensor_cut:
            if waveform.shape[1] > self.tensor_cut:
                start = torch.randint(0, waveform.shape[1] - self.tensor_cut, (1,))
                waveform = waveform[:, start : start + self.tensor_cut]

        return waveform, sample_rate


def load_audio_files(path, extension='.wav'):
    # Load the audio files
    audio_files = []
    for label in os.listdir(path):
        class_folder = os.path.join(path, label)
        if os.path.isdir(class_folder):
            for file in os.listdir(class_folder):
                if file.endswith(extension):
                    audio_files.append(os.path.join(path, label, file))
    return audio_files


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1)
    return batch


def collate_fn(batch):
    tensors = []

    for waveform, _ in batch:
        tensors += [waveform]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    return tensors