import torch
import torch.nn as nn

from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model

##Model Definition
class COVIDWav2Vec(nn.Module):
    def __init__(self, device):
        super(COVIDWav2Vec, self).__init__()
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.linear = nn.Linear(768,1)
        self.device = device
        return 

    def forward(self, input):
        tokenized = self.tokenizer(input, padding=True, return_tensors='pt').input_values.to(self.device)
        embedding = self.model(tokenized).last_hidden_state
        pooled = torch.mean(embedding, dim=1)
        linear_out = self.linear(pooled)
        return linear_out

