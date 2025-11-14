# CoFuncDesign
<img width="2327" height="1226" alt="image" src="https://github.com/user-attachments/assets/1e0b643e-25dc-4381-84a3-bc5bbc72fa23" />

# 🧬 **CoFuncDesign: Protein Sequence Design under Conserved Sequence and Functional Constraints**

**CoFuncDesign** is a unified framework for **de novo protein sequence generation** under both **functional** and **structural** constraints.  
It leverages **protein language models (PLMs)** in a dual-network architecture—a **generator** and an **evaluator**—to design novel protein sequences that meet specific biophysical or biochemical targets.




## 🧱 **Project Structure and Team Member Contributions**

**Lana Glisic:** Dataset Preprocessing and formatting
**Md Toki Tahmid:** Finetuning ESM and Sequence Generation
**Ravi Balasubramanian:** Evaluation and quality analysis of generated sequences
```bash
CoFuncDesign/
│
├── 📂 Codes/
│   │
│   ├── 📁 Finetuning/ **(Md Toki Tahmid)**
│   │   ├── finetune_dna_binding_site_prediction.py
│   │   ├── finetune_secondary_structure.py
│   │   └── finetune_solubility.py
│   │
│   ├── 📁 Generation/ **(Md Toki Tahmid)**
│   │   ├── generate_binding.py
│   │   ├── generate_sol.py
│   │   └── generate_ss.py
│   │
│   ├── 📁 Preprocessing/ **(Lana Glisic)**
│   │   ├── process_distance_map.py
│   │   └── processed_data.txt
│   │
│   └── 📁 Visualization/ **(Ravi Balasubramanian)**
│       ├── CoFuncDesign_Performance_Analysis.Rmd
│       ├── designed_sequences_binding_results.csv
│       ├── designed_sequences_solubility_results.csv
│       └── designed_sequences_ss.csv
│
├── 📂 Datasets/
│   └── 📁 Finetuning/
│       │
│       ├── 📁 DNA_binding_site_prediction/
│       │   ├── DNA-180-Test.fasta
│       │   └── DNA-735-Train.fasta
│       │
│       ├── 📁 secondary_structure/
│       │   └── data.csv
│       │
│       └── 📁 solvent_accessibility/
│           ├── asabu_training.csv
│           ├── asabu_validation.csv
│           └── asabu_test.csv
│
├── LICENSE
└── README.md
```
---

## 🚀 **Overview**

Understanding how amino acid sequences encode protein structure and function is a fundamental challenge in computational biology.  
While predictive models such as **AlphaFold2** and **ESM** learn mappings from *sequence → property*,  
**CoFuncDesign** addresses the *inverse problem*: generating new protein sequences that exhibit desired functional or structural properties.

CoFuncDesign integrates two independently fine-tuned **Evolutionary Scale Modeling (ESM)** networks:

- 🧩 **ESM-150M (Search Model)** — gradient-guided generator proposing candidate sequences.  
- 🧠 **ESM-650M (Evaluation Model)** — independent evaluator assessing property satisfaction.

Together, they form a **generator–evaluator feedback loop** that refines sequences toward user-defined biological objectives, balancing **novelty**, **accuracy**, and **interpretability**.

---

## ⚙️ **Methodology**

CoFuncDesign’s workflow consists of **four main stages**:

1. **🧾 Background Preparation**  
   Two PLMs (ESM-150M and ESM-650M) are fine-tuned for each target property—one for *generation* and one for *evaluation*—using curated datasets of protein sequences and property annotations.

2. **⚡ Gradient-Guided Generation**  
   A random amino acid sequence is iteratively updated by backpropagating gradients of the loss between predicted and desired property values.

3. **🧮 Evaluation and Stopping Criteria**  
   The independent evaluation model scores generated sequences periodically. Optimization halts when improvement plateaus.

4. **🔒 Sequence Constraints**  
   Conserved motifs or domains can be masked to preserve biological functionality. Optional probabilistic constraints allow flexible conservation of key residues.

---

## 🧩 **Target Properties**

CoFuncDesign currently supports **six major property types** — three structural and three functional:

| **Category** | **Property** | **Dataset** | **Format** | **Reference** |
|--------------|--------------|--------------|-------------|----------------|
| 🧱 *Structural* | **Secondary Structure** | [PS4 Dataset](https://github.com/omarperacha/ps4-dataset/tree/main/ps4_data/data) | Multiclass | Peracha *et al.*, 2024 |
|  | **Contact Map** | [CATH Dataset](https://www.cathdb.info/wiki/doku/?id=data:index#non-redundant_data_sets) | Binary | Sillitoe *et al.*, 2021 |
|  | **Distance Map** | [CATH Dataset](https://www.cathdb.info/wiki/doku/?id=data:index#non-redundant_data_sets) | Regression | Sillitoe *et al.*, 2021 |
| 🌿 *Functional* | **Solubility** | [SDBRNN](http://210.45.175.81:8080/rsa/sdbrnn.html) | Regression | Zhang *et al.*, 2018 |
|  | **DNA Binding** | [TransBind (DNA)](https://zenodo.org/records/10215073) | Binary | Tahmid *et al.*, 2025 |
|  | **RNA Binding** | [TransBind (RNA)](https://zenodo.org/records/10215073) | Binary | Tahmid *et al.*, 2025 |

---



