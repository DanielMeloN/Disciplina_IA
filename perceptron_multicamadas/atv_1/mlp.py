import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Data Loading directly from DOCX ---

def extract_tables_from_docx(docx_path):
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(docx_path) as docx:
        tree = ET.fromstring(docx.read('word/document.xml'))
        
    tables = []
    for tbl in tree.findall('.//w:tbl', ns):
        table_data = []
        for row in tbl.findall('.//w:tr', ns):
            row_data = []
            for cell in row.findall('.//w:tc', ns):
                texts = [t.text for t in cell.findall('.//w:t', ns) if t.text]
                row_data.append("".join(texts).strip())
            table_data.append(row_data)
        tables.append(table_data)
    return tables

# Caminho para o arquivo docx
docx_file = 'PMC1.docx'
if not os.path.exists(docx_file):
    # Tentar caminho absoluto caso rodado de outro lugar
    docx_file = '/home/alunos/Desktop/Disciplina_IA/perceptron_multicamadas/atv_1/PMC1.docx'

tables = extract_tables_from_docx(docx_file)

# Table 3 is the training set
train_table = tables[3]
train_data = []
for row in train_table[1:]: # Skip header
    # Parse in chunks of 5: Amostra, x1, x2, x3, d
    for i in range(0, len(row), 5):
        if i+4 < len(row) and row[i]:
            amostra = int(row[i])
            x1 = float(row[i+1])
            x2 = float(row[i+2])
            x3 = float(row[i+3])
            d = float(row[i+4])
            train_data.append([x1, x2, x3, d])

X_train = np.array([[row[0], row[1], row[2]] for row in train_data])
D_train = np.array([[row[3]] for row in train_data])

# Table 2 is the test set
test_table = tables[2]
test_data = []
for row in test_table[1:]: # Skip header
    if not row[0].isdigit():
        continue
    # Amostra, x1, x2, x3, d
    amostra = int(row[0])
    x1 = float(row[1])
    x2 = float(row[2])
    x3 = float(row[3])
    d = float(row[4])
    test_data.append([x1, x2, x3, d])

X_test = np.array([[row[0], row[1], row[2]] for row in test_data])
D_test = np.array([[row[3]] for row in test_data])

print(f"Train data size: {X_train.shape}, Test data size: {X_test.shape}")

# --- MLP Implementation (Pure NumPy, sem scikit-learn) ---

def sigmoid(x):
    # To prevent overflow
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(out):
    return out * (1.0 - out)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize weights with random values between 0 and 1
        # Include bias in weights
        self.W_hidden = np.random.rand(hidden_size, input_size + 1)
        self.W_out = np.random.rand(output_size, hidden_size + 1)
        
        self.lr = 0.1
        self.precision = 1e-6
        
    def train(self, X, D, max_epochs=100000):
        # Add bias to X
        N = X.shape[0]
        X_bias = np.hstack([np.ones((N, 1)), X])
        
        eqm_history = []
        epoch = 0
        
        while True:
            eqm_epoch = 0
            
            for i in range(N):
                x = X_bias[i:i+1].T # column vector
                d = D[i:i+1].T # column vector
                
                # Forward pass
                net_hidden = np.dot(self.W_hidden, x)
                out_hidden = sigmoid(net_hidden)
                
                # Add bias to hidden output
                out_hidden_bias = np.vstack([np.array([[1.0]]), out_hidden])
                
                net_out = np.dot(self.W_out, out_hidden_bias)
                out_final = sigmoid(net_out)
                
                # Error
                e = d - out_final
                eqm_epoch += e[0,0]**2
                
                # Backpropagation
                delta_out = e * sigmoid_derivative(out_final)
                
                # Hidden delta
                # Remove bias weight from W_out for backprop
                W_out_no_bias = self.W_out[:, 1:]
                delta_hidden = np.dot(W_out_no_bias.T, delta_out) * sigmoid_derivative(out_hidden)
                
                # Weight update
                self.W_out += self.lr * np.dot(delta_out, out_hidden_bias.T)
                self.W_hidden += self.lr * np.dot(delta_hidden, x.T)
                
            eqm_epoch /= N
            eqm_history.append(eqm_epoch)
            epoch += 1
            
            if epoch > 1:
                # Check precision
                if abs(eqm_history[-1] - eqm_history[-2]) <= self.precision:
                    break
            
            if epoch >= max_epochs:
                break
                
        return epoch, eqm_history
    
    def predict(self, X):
        N = X.shape[0]
        X_bias = np.hstack([np.ones((N, 1)), X])
        predictions = []
        for i in range(N):
            x = X_bias[i:i+1].T
            out_hidden = sigmoid(np.dot(self.W_hidden, x))
            out_hidden_bias = np.vstack([np.array([[1.0]]), out_hidden])
            out_final = sigmoid(np.dot(self.W_out, out_hidden_bias))
            predictions.append(out_final[0,0])
        return np.array(predictions)

# --- Execute Trainings ---

results = []
eqm_histories = []
models = []
seeds = [42, 100, 1234, 999, 777] # Ensure different initializations

if __name__ == '__main__':
    print("Starting training...")
    for i, seed in enumerate(seeds):
        print(f"Training T{i+1}...")
        mlp = MLP(input_size=3, hidden_size=10, output_size=1, seed=seed)
        epochs, eqm_hist = mlp.train(X_train, D_train)
        
        results.append({
            'Treinamento': f'T{i+1}',
            'EQM': eqm_hist[-1],
            'Epocas': epochs
        })
        eqm_histories.append((f'T{i+1}', eqm_hist))
        models.append(mlp)
        print(f"T{i+1}: Epocas={epochs}, EQM={eqm_hist[-1]:.6f}")

    # Find top 2 epochs
    sorted_by_epochs = sorted(results, key=lambda x: x['Epocas'], reverse=True)
    top2 = sorted_by_epochs[:2]

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, item in enumerate(top2):
        t_name = item['Treinamento']
        # Find history
        hist = next(h for n, h in eqm_histories if n == t_name)
        axes[i].plot(hist)
        axes[i].set_title(f'Treinamento {t_name} - EQM vs Épocas')
        axes[i].set_xlabel('Épocas')
        axes[i].set_ylabel('EQM')
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig('graficos_eqm.png')
    print("Saved plots to graficos_eqm.png")

    # --- Validation ---
    print("\nValidation Results:")
    test_predictions = []
    relative_errors = []
    variances = []

    for i, model in enumerate(models):
        preds = model.predict(X_test)
        test_predictions.append(preds)
        
        # Relative error: |d - y| / d * 100
        rel_error = np.abs(D_test.flatten() - preds) / D_test.flatten() * 100
        mean_rel_error = np.mean(rel_error)
        variance_rel_error = np.var(rel_error)
        
        relative_errors.append(mean_rel_error)
        variances.append(variance_rel_error)

    # Print test table
    print("Amostra", "d", *[f"yrede(T{i+1})" for i in range(5)])
    for i in range(len(D_test)):
        print(f"{i+1:2d}", f"{D_test[i][0]:.4f}", *[f"{test_predictions[j][i]:.4f}" for j in range(5)])

    print("\nErro Relativo Médio (%)", *[f"{re:.4f}" for re in relative_errors])
    print("Variância (%)", *[f"{va:.4f}" for va in variances])
