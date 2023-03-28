import torch
import torch.nn as nn

class TransformerAdder(nn.Module):
    def __init__(self, num_tokens, embedding_dim, num_layers):
        super(TransformerAdder, self).__init__()
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Define the embedding layer
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        
        # Define the transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Define the output layer
        self.output = nn.Linear(embedding_dim, 1)
        
    def forward(self, input_tokens):
        # Encode the input sequence using the transformer encoder
        embedded = self.embedding(input_tokens)
        encoded = self.encoder(embedded)
        
        # Compute the sum of the encoded sequence and return as output
        output = self.output(encoded.sum(dim=0))
        return output

# Create a toy dataset of addition problems
dataset = [(torch.tensor([i, j]), torch.tensor([i+j])) for i in range(10) for j in range(10)]

# Initialize the model and optimizer
model = TransformerAdder(num_tokens=10, embedding_dim=32, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(20):
    total_loss = 0
    for input_tokens, target_sum in dataset:
        model.zero_grad()
        output_sum = model(input_tokens)
        loss = ((output_sum - target_sum)**2).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Epoch %d, loss=%.4f" % (epoch, total_loss / len(dataset)))

# Test the model
test_data = [(torch.tensor([3, 4]), torch.tensor([7])),
             (torch.tensor([1, 8]), torch.tensor([9])),
             (torch.tensor([6, 2]), torch.tensor([8]))]
for input_tokens, target_sum in test_data:
    output_sum = model(input_tokens)
    print("Input tokens: %s, target sum: %d, output sum: %.2f" % (input_tokens.tolist(), target_sum.item(), output_sum.item()))
