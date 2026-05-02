# TeleNarratives

**TeleNarratives** is a research pipeline for detecting and analysing disinformation narratives in Ukrainian Telegram channels using graph neural networks and large language models.

The project spans the full data lifecycle — from raw Telegram message collection through narrative labelling, channel-forwarding graph construction, and supervised GNN classification.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Components](#pipeline-components)
- [Repository Structure](#repository-structure)
- [Notebooks](#notebooks)
- [Installation](#installation)
- [Configuration](#configuration)
- [Data](#data)
- [Acknowledgements](#acknowledgements)

---

## Overview

The pipeline processes Telegram messages from a curated set of Ukrainian-language channels and detects whether messages promote pre-defined disinformation narratives (e.g. *"Ukraine's victory is impossible"*, *"Discrediting Ukrainian officials"*). Classification is performed at two levels:

- **Strong labelling** — three human annotators manually label a shared subset (103 messages) and a golden set (412 messages); inter-annotator agreement is measured with Fleiss' κ and Krippendorff's α.
- **Weak labelling** — LLMs (GPT-4.1, Gemini 2.5 Flash, Claude) with ensemble voting, plus embedding similarity and multilingual NLI, assign labels to the remaining messages.
- **Graph-level reasoning** — a heterogeneous GNN over the channel-forwarding graph refines and propagates labels using structural signals.

---

## Pipeline Components

### 1. Data Parsing
Connects to the Telegram API via [Telethon](https://github.com/LonamiWebs/Telethon) and collects messages, channel metadata, and forward relationships for a configurable set of channels and time window.

**Entry point:** `src/disinfograph/parser.py`
**Notebook:** `notebooks/parse_messages.ipynb`

---

### 2. Data Cleaning and Preparation
Filters, deduplicates, and samples the raw message dump. Handles corrupt forward references and prepares the message frame for downstream labelling.

**Entry point:** `src/disinfograph/data_cleaner_and_sampler.py`
**Notebook:** `notebooks/data_prep_and_sampling.ipynb`

---

### 3. Data Annotation

#### 3a. Strong Labelling (Manual Human Annotation)
Three human annotators independently labeled a shared subset of **103 messages** to establish inter-annotator agreement, and a **golden set of 412 messages** used as the ground truth for downstream evaluation. Inter-annotator agreement was measured using Fleiss' κ and Krippendorff's α.

| File | Description |
|------|-------------|
| `src/disinfograph/labeling strategies/inter_annotator_agreement.py` | IAA computation (Fleiss' κ, Krippendorff's α) |
| `inter_annotator_agreement_metrics.csv` | IAA results |

#### 3b. Weak Labelling
Three automatic labelling strategies were implemented and compared against the human golden set to select the best-performing approach:

**LLM-based** — three LLMs independently assign a narrative label and confidence score; an ensemble majority-vote resolves disagreements.

| Script | Model |
|--------|-------|
| `src/disinfograph/labeling strategies/llms/gpt4.1.py` | GPT-4.1 |
| `src/disinfograph/labeling strategies/llms/gpt4o_mini.py` | GPT-4o mini |
| `src/disinfograph/labeling strategies/llms/gemini.py` | Gemini 2.5 Flash |
| `src/disinfograph/labeling strategies/llms/claude.py` | Claude |
| `src/disinfograph/labeling strategies/llms/ensemble_vote.py` | Majority-vote ensemble |

**Semantic similarity** — sentence embeddings (OpenAI, Gemini, HuggingFace `sentence-transformers`) propagate labels from labelled to unlabelled messages via nearest-neighbour similarity.

| Script | Method |
|--------|--------|
| `src/disinfograph/labeling strategies/sentence-transformers/embedding_openai.py` | OpenAI embeddings |
| `src/disinfograph/labeling strategies/sentence-transformers/embedding_gemini.py` | Gemini embeddings |
| `src/disinfograph/labeling strategies/sentence-transformers/embedding_hf.py` | HuggingFace embeddings |
| `src/disinfograph/labeling strategies/sentence-transformers/embedding_utils.py` | Shared similarity utilities |

**NLI-based** — zero-shot multilingual NLI (mDeBERTa-XNLI) classifies each message by treating narrative descriptions as hypotheses.

| Script | Method |
|--------|--------|
| `src/disinfograph/labeling strategies/multinli/mdeberta_xnli.py` | Zero-shot NLI classification |

All three strategies are evaluated against the human golden set in `src/disinfograph/labeling strategies/evaluate.py`.

---

### 4. Share Graph Construction and Analysis
Builds a heterogeneous channel-forwarding graph from the collected forward relationships. Nodes are channels; edges represent forwarding events weighted by frequency.

**Entry points:**
- `src/disinfograph/gnn/graph_builder.py` — NetworkX graph construction
- `src/disinfograph/gnn/neo4j_loader.py` — load to Neo4j
- `src/disinfograph/gnn/neo4j_export.py` — export from Neo4j into DGL

**Notebooks:** `notebooks/analyze_channel_forwards.ipynb`

---

### 5. Graph Training
A heterogeneous GNN is trained on the channel-forwarding graph to classify channels or messages into narrative categories. Includes a RoBERTa text baseline for comparison.

**Entry points:**
- `src/disinfograph/gnn/dgl.py` — DGL graph utilities and data loaders
- `src/disinfograph/gnn/heterographconv.py` — heterogeneous graph convolution layer
- `src/disinfograph/gnn/model.py` — GNN model definition
- `src/disinfograph/gnn/train_graph_model.py` — GNN training loop
- `src/disinfograph/gnn/train_roberta_text_baseline.py` — RoBERTa text baseline

**Notebook:** `notebooks/build_graph_train_gnn.ipynb`

> Note: notebooks 4 and 5 share the same notebook (`build_graph_train_gnn.ipynb`), which covers both graph construction and GNN training end-to-end.

---

## Repository Structure

```
TeleNarratives/
├── config/
│   ├── channels.csv            # Channel list with labels
│   ├── narratives.csv          # Narrative taxonomy
│   └── Important dates.csv     # Key event timeline
├── data/                       # Generated data files (see Data section)
├── notebooks/
│   ├── parse_messages.ipynb
│   ├── data_prep_and_sampling.ipynb
│   ├── analyze_channel_forwards.ipynb
│   └── build_graph_train_gnn.ipynb
├── src/disinfograph/
│   ├── config.py               # Centralised config (env vars, paths)
│   ├── parser.py               # Telegram data collection
│   ├── data_cleaner_and_sampler.py
│   ├── inter_annotator_agreement.py
│   ├── gnn/
│   │   ├── dgl.py
│   │   ├── graph_builder.py
│   │   ├── heterographconv.py
│   │   ├── model.py
│   │   ├── neo4j_export.py
│   │   ├── neo4j_loader.py
│   │   ├── train_graph_model.py
│   │   └── train_roberta_text_baseline.py
│   └── labeling strategies/
│       ├── evaluate.py
│       ├── inter_annotator_agreement.py
│       ├── llms/
│       │   ├── claude.py
│       │   ├── ensemble_vote.py
│       │   ├── gemini.py
│       │   ├── gpt4.1.py
│       │   └── gpt4o_mini.py
│       ├── multinli/
│       │   └── mdeberta_xnli.py
│       └── sentence-transformers/
│           ├── embedding_gemini.py
│           ├── embedding_hf.py
│           ├── embedding_openai.py
│           └── embedding_utils.py
├── gnn results/
├── labeling results/
├── requirements.txt
└── .env.example
```

---

## Notebooks

| Notebook | Pipeline step | Description |
|----------|--------------|-------------|
| `parse_messages.ipynb` | 1 | Collect messages and channel metadata from Telegram |
| `data_prep_and_sampling.ipynb` | 2 | Clean, filter, and sample raw messages |
| `analyze_channel_forwards.ipynb` | 4 | Build and explore the channel-forwarding graph |
| `build_graph_train_gnn.ipynb` | 4 + 5 | Construct DGL graph and train the GNN |

---

## Installation

```bash
git clone <repo-url>
cd TeleNarratives
pip install -r requirements.txt
```

The full pipeline was developed and run on a remote GPU server.

---

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:

| Variable | Description |
|----------|-------------|
| `TELEGRAM_API_ID` | From [my.telegram.org](https://my.telegram.org) |
| `TELEGRAM_API_HASH` | From [my.telegram.org](https://my.telegram.org) |
| `NEO4J_URI` | Neo4j connection URI (default: `bolt://localhost:7687`) |
| `NEO4J_USERNAME` | Neo4j username (default: `neo4j`) |
| `NEO4J_PASSWORD` | Neo4j password |
| `NEO4J_DATABASE` | Neo4j database name (default: `neo4j`) |
| `OPENAI_API_KEY` | For GPT-4 labelling and OpenAI embeddings |
| `ANTHROPIC_API_KEY` | For Claude labelling |
| `GEMINI_API_KEY` | For Gemini labelling and embeddings |

The channel list and narrative taxonomy are configured in `config/channels.csv` and `config/narratives.csv`.

---

## Data

The full dataset (raw messages, embeddings, labelled parquet files, and graph snapshots) is available on OSF:

> **OSF project:** *([link](https://osf.io/mjcs9/overview?view_only=07148b5ce53b412998ec179587ad8240))*

To reproduce the pipeline from scratch, configure your Telegram credentials and channel list, then run the notebooks in order.

---

## Acknowledgements

This work draws inspiration from the **MuMiN** project:

> Nielbo, K. L., Vermeer, S., Berber, H., & Winther, F. (2022).
> *MuMiN: A Large-Scale Multilingual Multimodal Fact-Checked Misinformation Social Network Dataset.*
> In Proceedings of ACL 2022.

MuMiN pioneered the use of heterogeneous social network graphs for misinformation detection, combining textual, visual, and structural signals. TeleNarratives adapts this graph-based paradigm to the Ukrainian Telegram ecosystem, focusing on narrative-level disinformation propagation through channel-forwarding networks.
