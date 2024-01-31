import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, trainCNN = False):
        super(EncoderCNN, self).__init__()

        self.trainCNN = trainCNN
        self.inception = models.inception_v3(pretrained=True, aux_logits = False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.inception(x)

        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.trainCNN

        return self.dropout(self.relu(x))
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
    
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, captions):

        embeddings = self.dropout(self.embedding(captions))
        embeddings = torch.cat((x.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        return self.linear(hiddens)

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()

        self.EncoderCNN = EncoderCNN(embed_size)
        self.DecoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, x, captions):
        features = self.EncoderCNN(x)
        return self.DecoderRNN(features, captions)
    
    def caption_image(self, image, vocabulary, max_length = 50):
        result = []

        with torch.no_grad():
            x = self.EncoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.DecoderRNN.lstm(x, states)
                output = self.DecoderRNN.linear(hiddens.unsqueeze(0))
                predicted = output.argmax(1)

                result.append(predicted.item())
                x = self.DecoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
            
        return [vocabulary.itos[idx] for idx in result]