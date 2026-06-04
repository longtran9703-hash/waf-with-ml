import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# Step 1: Ingest the Data
# ==========================================

# 1A. Load your malicious text files
with open('Auth_Bypass.txt', 'r', encoding='utf-8') as f:
    # Strip whitespace/newlines from each payload
    malicious_payloads = [line.strip() for line in f.readlines() if line.strip()]

df_malicious = pd.DataFrame({'payload': malicious_payloads, 'label': 1})

# 1B. Create or load Benign (Normal) data
# (If you have a normal.txt file, load it like above. Otherwise, here is a mock list)
normal_payloads = [
    "username=john_doe", "id=45", "search=sneakers", "page=about_us", 
    "category=electronics", "email=test@example.com", "sort_by=price_asc",
    "john.smith", "password123" # Add as many normal strings as possible
]
df_normal = pd.DataFrame({'payload': normal_payloads, 'label': 0})

# Combine and shuffle the dataset
df = pd.concat([df_malicious, df_normal]).sample(frac=1).reset_index(drop=True)
print(f"Total dataset size: {len(df)} rows")

# ==========================================
# Step 2: Feature Extraction (TF-IDF)
# ==========================================

print("Vectorizing payloads...")
# analyzer='char' is perfect for catching symbols like ' or 1=1--
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 4), max_features=3000)
X = vectorizer.fit_transform(df['payload']).toarray()
y = df['label'].values

# Save the vectorizer for FastAPI
joblib.dump(vectorizer, "../app/tokenizer.pkl")
print("Vectorizer saved to ../app/tokenizer.pkl")

# Convert to PyTorch Tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# ==========================================
# Step 3: Train the PyTorch MLP
# ==========================================

class WAFClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Outputs a probability between 0 and 1
        )
        
    def forward(self, x):
        return self.net(x)

# Initialize model, loss function, and optimizer
model = WAFClassifier(input_dim=3000)
criterion = nn.BCELoss() # Binary Cross Entropy for 0/1 classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training model...")
epochs = 15
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ==========================================
# Step 4: Export for FastAPI (TorchScript)
# ==========================================

model.eval()
# Create a dummy input to trace the model's execution path
dummy_input = torch.randn(1, 3000)
traced_model = torch.jit.trace(model, dummy_input)

traced_model.save("../models/waf_model.pt")
print("Model saved to ../models/waf_model.pt. Ready for FastAPI!")
