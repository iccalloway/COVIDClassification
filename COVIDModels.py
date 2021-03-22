import fairseq

import torch
import torch.nn as nn
import torchvision.models as models

##Model Definition
class COVIDWav2Vec(nn.Module):
    def __init__(self, device):
        super(COVIDWav2Vec, self).__init__()
        from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model, Wav2Vec2Config
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        config = Wav2Vec2Config(hidden_dropout_prob=0.25, attenion_probs_dropout_prob = 0.25)
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", config = config)
        self.linear = nn.Linear(768, 1)
        self.device = device
        return

    def forward(self, input):
        tokenized = self.tokenizer(
            input, padding=True, return_tensors="pt"
        ).input_values.to(self.device)
        embedding = self.model(tokenized).last_hidden_state
        pooled = torch.mean(embedding, dim=1)
        linear_out = self.linear(pooled)
        return linear_out

class COVIDFairseq(nn.Module):
    def __init__(self, device, path):
        super(COVIDFairseq, self).__init__()

        from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
        checkpoint = torch.load(path)
        config = Wav2Vec2Config()

        ##Custom Settings
        config.quantize_targets = True
        config.final_dim = 256
        
        model = Wav2Vec2Model.build_model(config)
        model.load_state_dict(checkpoint['model'])
        self.model = model
        self.linear = nn.Linear(512, 1)
        self.device = device
        return

    def forward(self, input):
        input = torch.unsqueeze(torch.from_numpy(input[0]), 0).to(torch.float).to(self.device)
        embedding = self.model.feature_extractor(input)
        pooled = torch.mean(embedding, dim=2)
        linear_out = self.linear(pooled)
        return linear_out

class DenseNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet, self).__init__()
        num_classes = 1 
        self.model = models.densenet201(pretrained=pretrained)
        self.model.classifier = nn.Linear(1920, num_classes) # 0, 1 if sigmoid(out)</> 0.5
        
    def forward(self, x):
        output = self.model(x)
        return output
