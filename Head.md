This is the explanation of what head function is kind of doing:

Let's go step by step and run **one iteration** of your `Head` function on the sentence:  

**"The cat sat on the mat"**  

Assuming:
- **`n_embd = 6`** (embedding dimension)
- **`head_size = 3`** (smaller split of `n_embd` for one attention head)
- **`block_size = 6`** (context size is the number of words)
- We assume **random embeddings** for simplicity.

---

### **1. Convert words into embeddings**
Each word is represented as a **6-dimensional vector** (since `n_embd=6`).  
For simplicity, let's assume these embeddings:

| Token  | Embedding (6D) |
|--------|---------------|
| The    | `[0.2, 0.5, -0.1, 0.3, 0.7, 0.0]` |
| cat    | `[0.1, -0.3, 0.4, -0.2, 0.6, 0.5]` |
| sat    | `[0.7, 0.2, -0.6, 0.1, 0.3, -0.4]` |
| on     | `[-0.2, 0.3, 0.5, 0.6, -0.1, 0.7]` |
| the    | `[0.5, -0.7, 0.2, 0.9, -0.3, 0.4]` |
| mat    | `[0.3, 0.1, -0.2, -0.5, 0.8, 0.6]` |

So our **input tensor `x`** is of shape **(1, 6, 6)** → **(Batch, Time Steps, Embedding Dim)**.

---

### **2. Apply Key, Query, Value linear layers**
Each **word embedding** is passed through three **linear layers** (with bias=False):

\[
K = x W_k, \quad Q = x W_q, \quad V = x W_v
\]

Assume **random weights** for `W_k`, `W_q`, and `W_v`:

\[
W_k =
\begin{bmatrix}
0.3 & 0.5 & -0.2 \\
-0.6 & 0.2 & 0.4 \\
0.1 & -0.3 & 0.5 \\
0.7 & -0.8 & 0.3 \\
-0.4 & 0.6 & -0.7 \\
0.2 & -0.1 & 0.8
\end{bmatrix}
\]

(Similar matrices for `W_q` and `W_v`).

Now, multiplying `x` with `W_k` gives **Key matrix (K)**:

\[
K =
\begin{bmatrix}
0.23 & -0.01 & 0.67 \\
-0.07 & 0.39 & 0.11 \\
0.49 & -0.36 & 0.42 \\
0.21 & 0.27 & 0.78 \\
0.55 & -0.48 & 0.31 \\
0.03 & 0.12 & 0.59
\end{bmatrix}
\]

(Similarly, we get **Q** and **V**).

---

### **3. Compute attention scores**
We calculate:

\[
\text{weights} = Q K^T / \sqrt{\text{head_size}}
\]

Since `head_size = 3`, we divide by \(\sqrt{3} \approx 1.732\).

Assume `QK^T` before scaling:

\[
\begin{bmatrix}
0.9 & -0.3 & 0.5 & 0.2 & 0.7 & -0.1 \\
-0.3 & 1.2 & -0.4 & 0.6 & 0.1 & 0.8 \\
0.5 & -0.4 & 1.0 & -0.2 & 0.9 & 0.3 \\
0.2 & 0.6 & -0.2 & 1.3 & 0.4 & 0.5 \\
0.7 & 0.1 & 0.9 & 0.4 & 1.1 & -0.3 \\
-0.1 & 0.8 & 0.3 & 0.5 & -0.3 & 1.4
\end{bmatrix}
\]

After scaling:

\[
\begin{bmatrix}
0.52 & -0.17 & 0.29 & 0.12 & 0.40 & -0.06 \\
-0.17 & 0.69 & -0.23 & 0.35 & 0.06 & 0.46 \\
0.29 & -0.23 & 0.58 & -0.12 & 0.52 & 0.17 \\
0.12 & 0.35 & -0.12 & 0.75 & 0.23 & 0.29 \\
0.40 & 0.06 & 0.52 & 0.23 & 0.63 & -0.17 \\
-0.06 & 0.46 & 0.17 & 0.29 & -0.17 & 0.81
\end{bmatrix}
\]

---

### **4. Apply mask (tril)**
We mask future tokens using:

```python
weights = weights.masked_fill(self.tril[:T,:T].bool() == 0, float('-inf'))
```

Resulting masked matrix:

\[
\begin{bmatrix}
0.52 & -\infty & -\infty & -\infty & -\infty & -\infty \\
-0.17 & 0.69 & -\infty & -\infty & -\infty & -\infty \\
0.29 & -0.23 & 0.58 & -\infty & -\infty & -\infty \\
0.12 & 0.35 & -0.12 & 0.75 & -\infty & -\infty \\
0.40 & 0.06 & 0.52 & 0.23 & 0.63 & -\infty \\
-0.06 & 0.46 & 0.17 & 0.29 & -0.17 & 0.81
\end{bmatrix}
\]

---

### **5. Apply softmax**
Softmax converts each row into probabilities:

\[
\begin{bmatrix}
1.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\
0.38 & 0.62 & 0.00 & 0.00 & 0.00 & 0.00 \\
0.41 & 0.22 & 0.37 & 0.00 & 0.00 & 0.00 \\
0.27 & 0.32 & 0.18 & 0.23 & 0.00 & 0.00 \\
0.31 & 0.17 & 0.27 & 0.14 & 0.11 & 0.00 \\
0.15 & 0.27 & 0.18 & 0.22 & 0.08 & 0.10
\end{bmatrix}
\]

Now, each token looks **only at past tokens**.

---

### **6. Compute weighted sum with V**
Finally, we do:

\[
\text{out} = \text{weights} \times V
\]

Each row is a weighted sum of the **value (V) embeddings**. This ensures words with higher attention contribute more.

---

### **7. Output**
The final **`out` matrix** is **(Batch, Time Steps, Head Size) = (1,6,3)**. This output is passed to the next layer in the Transformer.

---

### **Conclusion**
✔ **Attention scores are computed using Query and Key.**  
✔ **Masking prevents future tokens from being attended to.**  
✔ **Softmax converts scores into probabilities.**  
✔ **Weighted sum with Value determines final contextualized representation.**  

Now, the model knows how much each word influences another **before passing to the next Transformer block**.