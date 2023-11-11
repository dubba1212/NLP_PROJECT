# NER4ID at SemEval-2022 Task 2: Named Entity Recognition for Idiomaticity Detection

## Introduction
NER4ID is a system developed for the SemEval-2022 Task 2, focusing on Named Entity Recognition for Idiomaticity Detection. This project combines Transformer-based models and NER techniques to enhance the accuracy of idiomaticity detection in text.

## Citation
If you use our work or find it helpful, please cite our paper as follows:

```bibtex
@inproceedings{tedeschi-navigli-2022-ner4id,
    title = "{NER}4{ID} at {S}em{E}val-2022 Task 2: Named Entity Recognition for Idiomaticity Detection",
    author = "Simone Tedeschi and Roberto Navigli",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.25",
    doi = "10.18653/v1/2022.semeval-1.25",
    pages = "204--210",
    abstract = "Idioms are lexically-complex phrases whose meaning cannot be derived by compositionally interpreting their components. Although the automatic identification and understanding of idioms is essential for a wide range of Natural Language Understanding tasks, they are still largely under-investigated. This motivated the organization of the SemEval-2022 Task 2, which is divided into two multilingual subtasks: one about idiomaticity detection, and the other about sentence embeddings. In this work, we focus on the first subtask and propose a Transformer-based dual-encoder architecture to compute the semantic similarity between a potentially-idiomatic expression and its context and, based on this, predict idiomaticity. Then, we show how and to what extent Named Entity Recognition can be exploited to reduce the degree of confusion of idiom identification systems and, therefore, improve performance. Our model achieves 92.1 F1 in the one-shot setting and shows strong robustness towards unseen idioms achieving 77.4 F1 in the zero-shot setting. We release our code at https://github.com/Babelscape/ner4id.",
}
```

# NER4ID: Named Entity Recognition for Idiomaticity Detection

## Overview
NER4ID is an innovative approach to Named Entity Recognition (NER) focused on enhancing idiomaticity detection in text. This system integrates NER as a crucial component in the idiomaticity detection pipeline to improve accuracy and reduce errors.

## The NER4ID Model
### What is NER?
NER (Named Entity Recognition) is a key task in natural language processing involving the identification and categorization of named entities in text. These entities can range from person names, organizations, locations, to dates, and more. The main goal of NER is to convert unstructured text into structured data by identifying these entities and classifying them into predefined categories.

### Role in Idiomaticity Detection
In the NER4ID system, the NER module plays an auxiliary role, particularly in managing ambiguous cases. It helps in pre-identifying non-idiomatic expressions that are part of named entities, thereby enhancing the precision of idiomaticity detection. This is crucial in texts where expressions can be misinterpreted out of context.

## Data
We utilize datasets provided by SemEval-2022 Task 2 organizers, covering three languages: English, Portuguese, and Galician. These datasets feature multi-word expressions (MWEs) in different contexts to determine their literal or idiomatic usage. Our system is evaluated in two settings: zero-shot and one-shot, to ensure robust performance across varied data scenarios.

## Implementation
### System Architecture
The NER4ID model is built using PyTorch and the Transformers library, leveraging the power of a BERT-based model. Specifically, we employ the wikineural-multilingual-ner, an mBERT model fine-tuned on the WikiNEuRal dataset.

### Evaluation Metric
We adopt Macro F1 scores for comparing system performances, adhering to the standards set by the competition.

## Simplified Python Notebook
To facilitate understanding and usage, we have prepared a simplified Python Notebook. Key modifications include:
- **Unified Multilingual BERT Model**: A single BERT-base-multilingual-cased model is used for handling multiple languages.
- **Single Best Model Predictions**: Focus on the predictions from the best-performing model checkpoint.
- **SpaCy NER Tagger**: Integration of SpaCy's NER tagger for efficient entity recognition.

These changes are aimed at providing a user-friendly demonstration of NER4ID. The Python Notebook offers a detailed walkthrough and practical examples.

## Repository Contents
- Python Notebook: Detailed demonstration of the NER4ID model.
- Datasets: Multi-language datasets for idiomaticity detection.
- Pre-trained Models: Models used for NER and idiomaticity detection.


## Getting Started
To get started with NER4ID, clone this repository and follow the instructions in the Python Notebook.

## Acknowledgements
We thank the organizers of SemEval-2022 Task 2 and the contributors to the datasets and tools we used in this project.

## Additional Resources
- Official Paper: [NER4ID at SemEval-2022 Task 2](https://aclanthology.org/2022.semeval-1.25/)
- Code Repository: [GitHub - NER4ID](https://github.com/Babelscape/ner4id) 

