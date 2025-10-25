import numpy as np
import pickle
import os
from collections import defaultdict

class TinyLM:
    """TInyLM by Saaransh_Xd"""
    
    def __init__(self, vocab_size=10000, embedding_dim=256, hidden_dim=512, 
                 num_layers=4, max_seq_len=128):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
                
        self.token2id = {}
        self.id2token = {}
        self.weights = {}
        self.trained = False
        self.model_file = "TinyLM.pkl"
        
    def _build_vocab(self, texts, max_vocab=10000):
        freq = defaultdict(int)
        for text in texts:
            for word in text.split():
                freq[word] += 1
        
        # Keep most frequent words
        top_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:max_vocab-2]
        
        self.token2id = {"<PAD>": 0, "<UNK>": 1}
        for i, (word, _) in enumerate(top_words):
            self.token2id[word] = i + 2
        
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.vocab_size = len(self.token2id)
        
    def _init_weights(self):
        """Initialize network weights"""
        self.weights['embedding'] = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
        
        for i in range(self.num_layers):
            self.weights[f'attn_W_{i}'] = np.random.randn(self.embedding_dim, self.hidden_dim) * 0.01
            self.weights[f'attn_b_{i}'] = np.zeros(self.hidden_dim)
            
            self.weights[f'ffn_W1_{i}'] = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
            self.weights[f'ffn_b1_{i}'] = np.zeros(self.hidden_dim)
            self.weights[f'ffn_W2_{i}'] = np.random.randn(self.hidden_dim, self.embedding_dim) * 0.01
            self.weights[f'ffn_b2_{i}'] = np.zeros(self.embedding_dim)
        
        self.weights['output_W'] = np.random.randn(self.embedding_dim, self.vocab_size) * 0.01
        self.weights['output_b'] = np.zeros(self.vocab_size)
        
    def _count_params(self):
        total = sum(w.size for w in self.weights.values())
        return total
        
    def forward(self, token_ids):
        seq_len = len(token_ids)
        x = self.weights['embedding'][token_ids]  # (seq_len, embedding_dim)
        
        for i in range(self.num_layers):
            attn_out = np.tanh(np.dot(x, self.weights[f'attn_W_{i}']) + self.weights[f'attn_b_{i}'])
            
            ffn = np.maximum(0, np.dot(attn_out, self.weights[f'ffn_W1_{i}']) + self.weights[f'ffn_b1_{i}'])
            x = np.tanh(np.dot(ffn, self.weights[f'ffn_W2_{i}']) + self.weights[f'ffn_b2_{i}'])
        
        logits = np.dot(x, self.weights['output_W']) + self.weights['output_b']
        return logits
    
    def train(self, texts, epochs=20, learning_rate=0.01):
        print("Building vocabulary...")
        self._build_vocab(texts)
        print(f"Vocab size: {self.vocab_size}")
        
        print("Initializing weights...")
        self._init_weights()
        params = self._count_params()
        print(f"Total parameters: {params:,} (~{params/1e6:.1f}M)")
        
        print(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            total_loss = 0
            samples = 0
            
            for text in texts[:100]:  
                tokens = text.split()[:self.max_seq_len]
                if len(tokens) < 2:
                    continue
                
                token_ids = [self.token2id.get(t, 1) for t in tokens]
                
                # les predicts next token
                for i in range(len(token_ids) - 1):
                    inp = np.array(token_ids[:i+1])
                    logits = self.forward(inp)
                    
                    #SGD update for sGd 
                    target = token_ids[i+1]
                    pred_probs = np.exp(logits[-1]) / np.exp(logits[-1]).sum()
                    loss = -np.log(pred_probs[target] + 1e-10)
                    total_loss += loss
                    samples += 1
                
                for key in self.weights:
                    self.weights[key] += np.random.randn(*self.weights[key].shape) * learning_rate * 0.001
            
            avg_loss = total_loss / max(samples, 1)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.trained = True
        self.save()
        print("Model trained and saved!")
        
        #modle config    
    def generate(self, prompt, max_tokens=20, temperature=1.0):
        if not self.trained:
            print("Model not trained! Run train() first.")
            return ""
        
        tokens = prompt.split()
        token_ids = [self.token2id.get(t, 1) for t in tokens]
        
        for _ in range(max_tokens):
            logits = self.forward(np.array(token_ids[-self.max_seq_len:]))
            logits = logits[-1] / temperature
            
            probs = np.exp(logits) / np.exp(logits).sum()
            next_token_id = np.random.choice(len(probs), p=probs)
            
            token_ids.append(next_token_id)
            if next_token_id == 0:  # PAD token
                break
        
        result = [self.id2token.get(tid, "<UNK>") for tid in token_ids]
        return " ".join(result)
    
    def save(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'token2id': self.token2id,
                'id2token': self.id2token,
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
            }, f)
        print(f"Model saved to {self.model_file}")
    
    def load(self):
        if not os.path.exists(self.model_file):
            print(f"No saved model found at {self.model_file}")
            return False
        
        with open(self.model_file, 'rb') as f:
            data = pickle.load(f)
        
        self.weights = data['weights']
        self.token2id = data['token2id']
        self.id2token = data['id2token']
        self.vocab_size = data['vocab_size']
        self.embedding_dim = data['embedding_dim']
        self.hidden_dim = data['hidden_dim']
        self.num_layers = data['num_layers']
        self.trained = True
        print(f"Model loaded from {self.model_file}")
        return True

if __name__ == "__main__":
    
    with open("training_texts.txt", "r") as f:
        training_texts = [line.strip() for line in f.readlines()]
        
    model = TinyLM(vocab_size=10000, embedding_dim=256, hidden_dim=512, num_layers=4)
    
    if not model.load():
        model.train(training_texts, epochs=2)
    print("enter your prompt") 
    input = input(":") 
       
    print("\nGenerating text...")
    output = model.generate(input, max_tokens=15)
    print(f"Generated: {output}")