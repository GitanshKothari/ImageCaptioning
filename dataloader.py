import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class Vocabulary:
    def __init__(self, freq_thresh = 5):
        self.freq_thresh = freq_thresh
        self.itos = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>",
        }

        self.stoi = {
            "<PAD>": 0, 
            "<SOS>": 1, 
            "<EOS>": 2, 
            "<UNK>": 3,
        }

        self.spacy_eng = spacy.load('en_core_web_sm')

    def __len__(self):
        return len(self.itos)

    
    def tokenizer_eng(self, text):
        return [tok.text.lower() for tok in self.spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
            
            if frequencies[word] == self.freq_thresh:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    
    def get_ids(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, root, transform, freq_thresh = 5) -> None:
        super().__init__()
    
        self.root = root
        self.df = pd.read_csv(os.path.join(self.root, 'captions.txt'))
        self.transform = transform

        self.images = self.df["image"]
        self.captions = self.df["caption"]

        self.vocab = Vocabulary(freq_thresh)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_name = self.images[index]

        img = Image.open(os.path.join(self.root, 'Images', img_name)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        
        caption_ids = [self.vocab.stoi["<SOS>"]]
        caption_ids += self.vocab.get_ids(caption)
        caption_ids.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(caption_ids)
    
    @staticmethod
    def collate_fn(batch, pad_idx):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=pad_idx)

        return imgs, targets

def get_loader(root, transform, batch_size=32, num_workers=0, shuffle=True, pin_memory=True):
    dataset = FlickrDataset(root, transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
        pin_memory = pin_memory,
        collate_fn = lambda batch: FlickrDataset.collate_fn(batch, pad_idx)
        )
    
    return loader, dataset

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataloader, dataset = get_loader(root = r'C:\Users\gitakoth\personal_projects\ImageCaptioning\Flickr8k', transform=transform)

    for _, (imgs, captions) in enumerate(dataloader):
        print(imgs.shape)
        print(captions.shape)

if __name__ == '__main__':
    main()