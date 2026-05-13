# Resolução da Atividade 1 - Perceptron Multicamadas

## 1. Treinamentos da Rede Perceptron

**Atividade:** *Execute 5 treinamentos para a rede PERCEPTRON inicializando as matrizes de pesos em cada treinamento com valores aleatórios entre 0 e 1... Registre os resultados finais desses 5 treinamentos.*

A rede foi treinada 5 vezes (denominadas T1 a T5), inicializando os pesos sinápticos aleatoriamente entre 0 e 1 em cada execução. Foi utilizada a função de ativação logística em todos os neurônios, taxa de aprendizado $\eta = 0.1$ e precisão de parada $\epsilon = 10^{-6}$.

**Tabela de Resultados dos Treinamentos:**

| Treinamento | Erro Quadrático Médio (EQM) | Número de Épocas |
| :--- | :--- | :--- |
| 1º (T1) | 0.001575 | 126 |
| 2º (T2) | 0.001590 | 137 |
| 3º (T3) | 0.001597 | 158 |
| 4º (T4) | 0.001574 | 148 |
| 5º (T5) | 0.001588 | 155 |

---

## 2. Gráficos de Erro Quadrático Médio (EQM)

**Atividade:** *Para os dois treinamentos com maiores números de épocas, trace os respectivos gráficos dos valores de erro quadrático médio (EQM) em função de cada época de treinamento.*

Os dois treinamentos que demandaram o maior número de épocas para atingir o critério de convergência foram o **T3 (158 épocas)** e o **T5 (155 épocas)**.

![Gráficos de EQM vs Épocas](graficos_eqm.png)

*(Os gráficos mostram uma queda acentuada do erro nas primeiras épocas, estabilizando-se assintoticamente até que a variação do erro entre épocas atinja a precisão exigida de $10^{-6}$).*

---

## 3. Variação de EQM e Épocas

**Atividade:** *Baseado na tabela, explique de forma detalhada por que tanto o erro quadrático médio quanto o número de épocas variam de treinamento para treinamento.*

**Resposta:**
Esta variação ocorre estritamente devido à **inicialização aleatória das matrizes de pesos sinápticos**. O algoritmo de *Backpropagation* (Regra Delta Generalizada) é um método de otimização baseado em gradiente descendente, cujo objetivo é encontrar o mínimo global (ou um mínimo local satisfatório) na superfície topológica do erro da rede neural.

Ao atribuir valores iniciais aleatórios e diferentes aos pesos em cada treinamento, a rede "inicia sua caminhada" a partir de um ponto coordenado diferente nessa superfície de erro multidimensional. Consequentemente:
1. **Variação de Épocas:** A distância e o caminho (inclinação do gradiente) que o algoritmo precisa percorrer até encontrar um vale que satisfaça a precisão ($\Delta EQM \leq 10^{-6}$) são diferentes a cada tentativa.
2. **Variação do EQM:** A rede frequentemente converge para diferentes mínimos locais na superfície de erro, resultando em valores finais de EQM distintos para cada inicialização.

---

## 4. Validação da Rede (Conjunto de Teste)

**Atividade:** *Faça a validação da rede aplicando o conjunto de teste fornecido... Forneça para cada treinamento o erro relativo médio (%) e a respectiva variância.*

A rede validou as amostras não vistas durante o treinamento. A tabela abaixo condensa os valores preditos $y_{rede}$ e as estatísticas comparativas em relação aos valores desejados $d$:

| Amostra | $d$ (desejado) | $y_{rede}$(T1) | $y_{rede}$(T2) | $y_{rede}$(T3) | $y_{rede}$(T4) | $y_{rede}$(T5) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.4831 | 0.4970 | 0.4961 | 0.4981 | 0.4933 | 0.4946 |
| 2 | 0.5965 | 0.6038 | 0.6021 | 0.6041 | 0.6025 | 0.6040 |
| 3 | 0.5318 | 0.5381 | 0.5380 | 0.5377 | 0.5349 | 0.5371 |
| 4 | 0.6843 | 0.7167 | 0.7169 | 0.7166 | 0.7176 | 0.7161 |
| 5 | 0.2872 | 0.2792 | 0.2841 | 0.2792 | 0.2861 | 0.2820 |
| 6 | 0.7663 | 0.7603 | 0.7627 | 0.7617 | 0.7616 | 0.7616 |
| 7 | 0.5666 | 0.5817 | 0.5813 | 0.5838 | 0.5787 | 0.5812 |
| 8 | 0.6601 | 0.6920 | 0.6918 | 0.6927 | 0.6905 | 0.6911 |
| 9 | 0.5427 | 0.5450 | 0.5423 | 0.5444 | 0.5420 | 0.5444 |
| 10 | 0.5836 | 0.6165 | 0.6164 | 0.6194 | 0.6128 | 0.6157 |
| 11 | 0.6950 | 0.7024 | 0.7026 | 0.7027 | 0.7023 | 0.7015 |
| 12 | 0.6790 | 0.6857 | 0.6854 | 0.6858 | 0.6858 | 0.6852 |
| 13 | 0.2956 | 0.2871 | 0.2939 | 0.2878 | 0.2925 | 0.2909 |
| 14 | 0.7742 | 0.7883 | 0.7888 | 0.7892 | 0.7902 | 0.7879 |
| 15 | 0.4662 | 0.4754 | 0.4744 | 0.4769 | 0.4727 | 0.4731 |
| 16 | 0.8093 | 0.8204 | 0.8238 | 0.8207 | 0.8241 | 0.8224 |
| 17 | 0.7581 | 0.7848 | 0.7865 | 0.7851 | 0.7889 | 0.7852 |
| 18 | 0.5826 | 0.6025 | 0.6021 | 0.6025 | 0.6012 | 0.6021 |
| 19 | 0.7938 | 0.8033 | 0.8049 | 0.8029 | 0.8057 | 0.8028 |
| 20 | 0.5012 | 0.5082 | 0.5067 | 0.5100 | 0.5048 | 0.5060 |
| **Erro Relativo Médio (%)** | | **2.3397** | **2.0904** | **2.4144** | **1.9645** | **2.1052** |
| **Variância (%)** | | **2.0920** | **2.4077** | **2.3258** | **2.2952** | **2.1101** |

---

## 5. Conclusão da Melhor Configuração

**Atividade:** *Indique qual das configurações finais de treinamento seria a mais adequada para o sistema de ressonância magnética, ou seja, qual delas está oferecendo a melhor generalização.*

**Resposta:**
A configuração mais adequada é a do **Treinamento 4 (T4)**. 

O principal objetivo do treinamento é fazer com que a rede generalize bem (evitando *overfitting* aos dados de treino). Analisando a tabela de validação, observa-se que o modelo T4 obteve o menor **Erro Relativo Médio (1.9645%)** sobre o conjunto de testes (dados cegos). Uma taxa menor de erro médio nas amostras não vistas comprova estatisticamente que a configuração T4 conseguiu "aprender" o comportamento subjacente da variável resposta $y$ da melhor forma, oferecendo assim o aproximador mais robusto e fidedigno para a operação do sistema de ressonância magnética no mundo real.

---

## Anexo: Código-Fonte em Python (mlp.py)

O script abaixo foi desenvolvido de forma a **não utilizar frameworks prontos (como Scikit-Learn)**, realizando o processamento matricial do algoritmo diretamente com a biblioteca nativa `numpy`. Ele lê os dados diretamente do arquivo DOCX providenciado.

```python
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
```
