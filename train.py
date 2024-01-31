import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from model import CNNtoRNN
from dataloader import get_loader
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train():
    # Hyper-parameters
    embed_size = 256
    hidden_size = 512
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100
    batch_size = 32
    vocab_threshold = 5

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_loader, dataset = get_loader(
        root='flickr8k/images',
        transform=transform,
        batch_size=batch_size,
    )

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        pbar = tqdm(enumerate(train_loader), desc='Epoch: {}'.format(epoch))
        for idx, (imgs, captions) in pbar:
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Forward prop
            outputs = model(imgs, captions[:-1])

            # Calculate loss
            targets = captions[1:]
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))

            # Backward prop
            optimizer.zero_grad()
            loss.backward()

            # Update model
            optimizer.step()

            pbar.set_postfix({'loss': loss.item()})
            pbar.update()

    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, 'checkpoint.pth')
    
if __name__ == '__main__':
    train()

            