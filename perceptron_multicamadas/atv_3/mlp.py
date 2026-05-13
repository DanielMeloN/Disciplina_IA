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

docx_file = 'PMC3.docx'
if not os.path.exists(docx_file):
    docx_file = '/home/alunos/Desktop/Disciplina_IA/perceptron_multicamadas/atv_3/PMC3.docx'

tables = extract_tables_from_docx(docx_file)

# We have 120 points in total. t=1..100 in table 3, t=101..120 in table 2.
F = np.zeros(120)

# Parse table 3 (Train: 1 to 100)
for row in tables[3][1:]: # Skip header
    for i in range(0, len(row), 2):
        if i+1 < len(row) and row[i].startswith('t ='):
            t_val = int(row[i].replace('t =', '').strip())
            f_val = float(row[i+1].replace(',', '.'))
            F[t_val - 1] = f_val

# Parse table 2 (Test: 101 to 120)
for row in tables[2][2:]: # Skip two header rows
    if row[0].startswith('t ='):
        t_val = int(row[0].replace('t =', '').strip())
        f_val = float(row[1].replace(',', '.'))
        F[t_val - 1] = f_val

def create_dataset(series, p, t_start, t_end):
    # t_start and t_end are 1-indexed (e.g., 1 to 100)
    # Target index in array: t-1
    # Inputs: f(t-1), f(t-2), ..., f(t-p) -> array indices: t-2, t-3, ..., t-1-p
    X = []
    D = []
    for t in range(t_start, t_end + 1):
        idx = t - 1
        # Inputs: [F[idx-1], F[idx-2], ..., F[idx-p]]
        x = [series[idx - k] for k in range(1, p + 1)]
        X.append(x)
        D.append([series[idx]])
    return np.array(X), np.array(D)

# --- MLP Implementation ---

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(out):
    return out * (1.0 - out)

class TDNN:
    def __init__(self, p, hidden_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.W_hidden = np.random.rand(hidden_size, p + 1)
        self.W_out = np.random.rand(1, hidden_size + 1)
        
        self.lr = 0.1
        self.precision = 0.5e-6
        self.momentum = 0.8
        
    def train(self, X, D, max_epochs=100000):
        N = X.shape[0]
        X_bias = np.hstack([np.ones((N, 1)), X])
        
        eqm_history = []
        epoch = 0
        
        v_W_out = np.zeros_like(self.W_out)
        v_W_hidden = np.zeros_like(self.W_hidden)
        
        while True:
            eqm_epoch = 0
            
            for i in range(N):
                x = X_bias[i:i+1].T 
                d = D[i:i+1].T 
                
                # Forward
                net_hidden = np.dot(self.W_hidden, x)
                out_hidden = sigmoid(net_hidden)
                
                out_hidden_bias = np.vstack([np.array([[1.0]]), out_hidden])
                
                net_out = np.dot(self.W_out, out_hidden_bias)
                out_final = sigmoid(net_out)
                
                # Error
                e = d - out_final
                eqm_epoch += e[0,0]**2
                
                # Backprop
                delta_out = e * sigmoid_derivative(out_final)
                
                W_out_no_bias = self.W_out[:, 1:]
                delta_hidden = np.dot(W_out_no_bias.T, delta_out) * sigmoid_derivative(out_hidden)
                
                grad_W_out = np.dot(delta_out, out_hidden_bias.T)
                grad_W_hidden = np.dot(delta_hidden, x.T)
                
                v_W_out = self.momentum * v_W_out + self.lr * grad_W_out
                v_W_hidden = self.momentum * v_W_hidden + self.lr * grad_W_hidden
                
                self.W_out += v_W_out
                self.W_hidden += v_W_hidden
                
            eqm_epoch /= N
            eqm_history.append(eqm_epoch)
            epoch += 1
            
            if epoch > 1:
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

if __name__ == '__main__':
    topologies = [
        {'name': 'Rede 1', 'p': 5, 'N1': 10},
        {'name': 'Rede 2', 'p': 10, 'N1': 15},
        {'name': 'Rede 3', 'p': 15, 'N1': 25}
    ]
    seeds = [42, 100, 1234]
    
    results = {}
    best_models = {}
    
    print("Starting trainings...")
    
    for topo in topologies:
        name = topo['name']
        p = topo['p']
        N1 = topo['N1']
        results[name] = []
        
        # Train dataset: t = p+1 to 100
        X_train, D_train = create_dataset(F, p, p + 1, 100)
        # Test dataset: t = 101 to 120
        X_test, D_test = create_dataset(F, p, 101, 120)
        
        best_error = float('inf')
        best_model_info = None
        
        for i, seed in enumerate(seeds):
            t_name = f'T{i+1}'
            print(f"Training {name} - {t_name}...")
            net = TDNN(p, N1, seed=seed)
            epochs, eqm_hist = net.train(X_train, D_train)
            
            preds = net.predict(X_test)
            rel_errors = np.abs(D_test.flatten() - preds) / D_test.flatten() * 100
            mean_rel_error = np.mean(rel_errors)
            var_rel_error = np.var(rel_errors)
            
            res = {
                'T': t_name,
                'EQM': eqm_hist[-1],
                'Epochs': epochs,
                'Mean_Rel_Error': mean_rel_error,
                'Var_Rel_Error': var_rel_error,
                'Preds': preds,
                'EQM_Hist': eqm_hist,
                'D_test': D_test.flatten()
            }
            results[name].append(res)
            print(f"  {t_name}: Epochs={epochs}, EQM={eqm_hist[-1]:.6f}, Mean Error={mean_rel_error:.2f}%")
            
            if mean_rel_error < best_error:
                best_error = mean_rel_error
                best_model_info = res
                
        best_models[name] = best_model_info

    # Plot 1: EQM vs Epochs for the best training of each network
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
    for idx, name in enumerate(['Rede 1', 'Rede 2', 'Rede 3']):
        bm = best_models[name]
        axes1[idx].plot(bm['EQM_Hist'], color='blue')
        axes1[idx].set_title(f'{name} ({bm["T"]}) - EQM vs Épocas')
        axes1[idx].set_xlabel('Épocas')
        axes1[idx].set_ylabel('EQM')
        axes1[idx].grid(True)
    plt.tight_layout()
    plt.savefig('grafico_eqm.png')
    
    # Plot 2: Desired vs Estimated for the best training of each network
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    t_vals = np.arange(101, 121)
    for idx, name in enumerate(['Rede 1', 'Rede 2', 'Rede 3']):
        bm = best_models[name]
        axes2[idx].plot(t_vals, bm['D_test'], marker='o', label='Desejado')
        axes2[idx].plot(t_vals, bm['Preds'], marker='x', label='Estimado')
        axes2[idx].set_title(f'{name} ({bm["T"]}) - Previsão')
        axes2[idx].set_xlabel('t')
        axes2[idx].set_ylabel('f(t)')
        axes2[idx].legend()
        axes2[idx].grid(True)
    plt.tight_layout()
    plt.savefig('grafico_previsao.png')
    
    print("\nTraining completed. Plots saved.")
    
    # Dump table results for markdown construction
    print("\n--- REPORT DATA ---")
    print("Table 1: EQM and Epochs")
    for name in ['Rede 1', 'Rede 2', 'Rede 3']:
        for res in results[name]:
            print(f"{name} {res['T']}: EQM={res['EQM']:.6f}, Epochs={res['Epochs']}")
            
    print("\nTable 2: Test Set Validation")
    for name in ['Rede 1', 'Rede 2', 'Rede 3']:
        for res in results[name]:
            print(f"{name} {res['T']}: Err={res['Mean_Rel_Error']:.4f}%, Var={res['Var_Rel_Error']:.4f}%")
            print(f"  Preds: {[round(x,4) for x in res['Preds']]}")
            
    # Calculate global best
    global_best = min(best_models.items(), key=lambda x: x[1]['Mean_Rel_Error'])
    print(f"\nBest Model: {global_best[0]} - {global_best[1]['T']} with Err={global_best[1]['Mean_Rel_Error']:.4f}%")
