# CoFuncDesign
<img width="2327" height="1226" alt="image" src="https://github.com/user-attachments/assets/1e0b643e-25dc-4381-84a3-bc5bbc72fa23" />



🧬 CoFuncDesign: Protein Sequence Design under Conserved Sequence and Functional Constraints

CoFuncDesign is a generalized framework for de novo protein sequence generation under both functional and structural constraints.
It builds on protein language models (PLMs) and introduces a dual-model optimization loop—a generator and an evaluator—to design protein sequences that satisfy desired biological properties.

🚀 Overview

Understanding how amino acid sequences encode protein structure and function is a central challenge in computational biology.
While predictive models like AlphaFold2 and ESM learn mappings from sequence → property, CoFuncDesign tackles the inverse problem: generating new protein sequences given desired structural or functional profiles.

CoFuncDesign integrates two independently fine-tuned PLMs:

ESM-150M (Search Model) – gradient-guided generator that proposes new sequences.

ESM-650M (Evaluation Model) – independent, unbiased evaluator that scores sequences on the target property.

This generator–evaluator loop enables iterative optimization of sequences toward user-defined objectives while maintaining biological plausibility and interpretability.

⚙️ Methodology

The workflow consists of four main components:

Background Preparation
Fine-tune two ESM models on curated datasets for each target property—one for generation, one for evaluation.

Gradient-Guided Generation
Initialize a random sequence and iteratively modify residues using backpropagated gradients that minimize deviation from the target property.

Evaluation and Stopping Criteria
Periodically assess generated sequences using the independent evaluation model; stop when improvement plateaus.

Sequence Constraints
Enforce conservation by masking functionally important motifs or domains during optimization.

🧩 Target Properties

CoFuncDesign currently supports six major structural and functional property constraints:

Category	Property	Dataset	Format	Reference
Structural	Secondary Structure	PS4 Dataset
	Multiclass	Peracha et al., 2024
	Contact Map	CATH Dataset
	Binary	Sillitoe et al., 2021
	Distance Map	CATH Dataset
	Regression	Sillitoe et al., 2021
Functional	Solubility	SDBRNN
	Regression	Zhang et al., 2018
	DNA Binding	TransBind (DNA)
	Binary	Tahmid et al., 2025
	RNA Binding	TransBind (RNA)
	Binary	Tahmid et al., 2025
🧠 Relation to Course Concepts

This project extends protein language modeling and structural property prediction by inverting the conventional prediction direction—moving from
sequence → property to property → sequence generation.
It combines model fine-tuning, gradient-based optimization, and independent validation using established computational biology pipelines.

📊 Evaluation Metrics

Generated sequences are evaluated based on:

Target Alignment – performance under the independent evaluator.

Fidelity – cosine similarity between embeddings of real and generated sequences.

Diversity – n-gram or pairwise sequence dissimilarity.

Robustness – correlation between evaluator predictions before and after optimization.

📂 Project Structure
CoFuncDesign/
├── figures/
│   ├── COS551_Proposal.pdf
│   ├── solubility_distribution.png
│   ├── dna_binding_distribution.png
│   └── ss_labels_distribution.png
├── ref.bib
└── CoFuncDesign_Proposal.tex

👥 Contributors
Name	Department	Year
Md Toki Tahmid	Computer Science	1st-Year PhD
Ravi Balasubramanian	Quantitative & Computational Biology	1st-Year PhD
Lana Glisic	Computer Science	1st-Year MS
📚 Key References

Rives et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. PNAS.

Madani et al. (2020). ProGen: Language modeling for protein generation. Nature Biotechnology.

Watson et al. (2023). De novo design of proteins using diffusion models. Nature.

Shahgir et al. (2024). RNA-DCGen: Dual-model RNA sequence generation framework. bioRxiv.
