Official repository for the paper [NER4ID at SemEval-2022 Task 2: Named Entity Recognition for Idiomaticity Detection](https://www.researchgate.net/publication/360541089_NER4ID_at_SemEval-2022_Task_2_Named_Entity_Recognition_for_Idiomaticity_Detection).

--------------------------------------------------------------------------------

Citation:

#### Bibtex
```bibtex
@inproceedings{tedeschi-navigli-2022-ner4id,
    title = "{NER}4{ID} at {S}em{E}val-2022 Task 2: Named Entity Recognition for Idiomaticity Detection",
    author = "Tedeschi, Simone  and
      Navigli, Roberto",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.25",
    doi = "10.18653/v1/2022.semeval-1.25",
    pages = "204--210",
    abstract = "Idioms are lexically-complex phrases whose meaning cannot be derived by compositionally interpreting their components. Although the automatic identification and understanding of idioms is essential for a wide range of Natural Language Understanding tasks, they are still largely under-investigated.This motivated the organization of the SemEval-2022 Task 2, which is divided into two multilingual subtasks: one about idiomaticity detection, and the other about sentence embeddings. In this work, we focus on the first subtask and propose a Transformer-based dual-encoder architecture to compute the semantic similarity between a potentially-idiomatic expression and its context and, based on this, predict idiomaticity. Then, we show how and to what extent Named Entity Recognition can be exploited to reduce the degree of confusion of idiom identification systems and, therefore, improve performance.Our model achieves 92.1 F1 in the one-shot setting and shows strong robustness towards unseen idioms achieving 77.4 F1 in the zero-shot setting. We release our code at https://github.com/Babelscape/ner4id.",
}
```
<br>

# The NER4ID model
NER, or Named Entity Recognition, is a natural language processing task that involves identifying and categorizing named entities in text. These entities can include person names, organization names, locations, dates, and more. The purpose of NER is to extract structured information from unstructured text data.
The NER module serves as an auxiliary component in the idiomaticity detection pipeline. It helps manage ambiguous cases by pre-identifying non-idiomatic expressions that are part of named entities. These expressions, although they may appear in the text, may be unrelated to the context in which they are used. By identifying these expressions, the NER module prevents errors in the idiomaticity detection process.
It takes a raw text sequence containing a potentially idiomatic expression and predicts all the entities present in the sequence. It accomplishes this by assigning predefined semantic types (e.g., Person, Location, Organization) to specific words, thus identifying them as belonging to those types.
Overall, NER is a valuable tool for extracting structured information from unstructured text data. It enhances the accuracy of the idiomaticity detection system by effectively managing ambiguous cases.

# Data
The datasets used to train and evaluate our NER4ID system are those provided by [SemEval-2022 Task 2](https://sites.google.com/view/semeval2022task2-idiomaticity) organizers. Each entry contains a multi-word expression (MWE) in context, and the aim of the system is to determine whether such MWE is used with a literal or idiomatic meaning in that context. Datasets are provided for three different languages: English, Portuguese and Galician.

Additionally, two different settings are available: zero-shot and one-shot.
In the "zero-shot" setting, MWEs (potentially idiomatic phrases) in the training set are completely disjoint from those in the test and development sets. In the "one-shot" setting, they included one positive and one negative training example for each MWE in the test and development sets.

The datasets which we used are available in the github repository

<br>

# Implementation
We implemented our idiom identification system with [PyTorch](https://pytorch.org/) using the [Transformers library](https://huggingface.co/docs/transformers/index) to load the weights of a BERT-based model.
To identify entities, instead, we employed [wikineural-multilingual-ner](https://huggingface.co/Babelscape/wikineural-multilingual-ner), a Multilingual BERT (mBERT) model fine-tuned on the [WikiNEuRal](https://github.com/babelscape/wikineural) dataset. We compare systems by means of their Macro F1 scores, as specified by the competition rules.

We have prepared a Python Notebook that demonstrates the key modules of the NER4ID system. To make it more user-friendly, we have simplified the notebook with the following modifications:

Unified Multilingual BERT Model: Instead of utilizing separate BERT models like BERT-base-cased for English and BERT-base-portuguese-cased for Portuguese and Galician, we employ a single BERT model known as BERT-base-multilingual-cased. This model can handle multiple languages, allowing for a streamlined approach.

Single Best Model Predictions: Instead of ensembling the predictions from 9 different model checkpoints, we focus solely on the predictions of the best-performing model. This simplifies the process and eliminates the need for combining multiple model outputs.

SpaCy NER Tagger: To identify entities, we utilize the widely-used SpaCy NER tagger. This tagger is known for its effectiveness in recognizing named entities in text, ensuring accurate entity identification within the NER4ID system.

By implementing these simplifications, we aim to provide a more accessible and straightforward demonstration of the NER4ID system. Please refer to the Python Notebook for a detailed walkthrough and practical examples.
