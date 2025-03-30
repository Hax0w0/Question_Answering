# Question Answering (Generative vs Classification) README
 **Project**: Question Answering<br>
 **Class**: Northwestern CS 461 Winter 2025<br>
 **Contributers**: Raymond Gu, Maanvi Sarwadi

## OpenBookQA Dataset
`OpenBookQA` is a multiple-choice question-answering dataset of elementary school-level scientific questions. Questions are posed in natural language with four possible choices. Background knowledge in the form of a one-sentence “fact” is also provided. For question answering, we would usually train the model to identify which fact should be used to answer each question. However, to focus on transformer architectures, we are given the fact associated with each question.
- **Stem**: This is the question that the model is trying to find the answer for.<br>
- **Fact**: This is information that is needed to answer the question.<br>
- **Choices**: Each of the choices have a label (A, B, C, or D) and a possible answer to the question.<br>
- **Answer Key**: The correct choice among the provided options (A, B, C, or D).<br><br>

## Generative Model
The `Generative.py` file contains all the code needed for our generative model.<br>

**Generative Approach Overview**<br>
For our generative approach, we decided to encode each instance in the training set in the format shown below: <br>

> [START] \<Fact\> \<Stem\> [A] \<Option_A\> [B] \<Option_B\> [C] \<Option_C\> [D] \<Option_D\> [ANSWER] \<Answer_Key\>

Each instance in the validation and test sets are encoded in the format shown below: <br>

> [START] \<Fact\> \<Stem\> [A] \<Option_A\> [B] \<Option_B\> [C] \<Option_C\> [D] \<Option_D\> [ANSWER]

For our generative model, we used `GPT2LMHeadModel`, which has a decoder-only architecture, and `GPT2Tokenizer`. We would typically use GPT-2 to predict the next word for each word in the sequence. However, for our approach, we only compute the loss for the final answer token. By doing this, we are training the model to map the entire input sequence to an answer choice (A, B, C, or D).<br><br>

## Classification Model
The `Classification.py` file contains all the code needed for our classification model.<br>

**Classification Approach Overview**<br>
For our classification approach, we encoded each instance in the dataset in the format shown below: <br>

> [CLS] \<Fact\> [SEP] \<Stem\> [SEP] \<Option_A\> [SEP] <br>
> [CLS] \<Fact\> [SEP] \<Stem\> [SEP] \<Option_B\> [SEP] <br>
> [CLS] \<Fact\> [SEP] \<Stem\> [SEP] \<Option_C\> [SEP] <br>
> [CLS] \<Fact\> [SEP] \<Stem\> [SEP] \<Option_D\> [SEP] <br>

We used the bert-base-uncased checkpoint for the `BertModel` and `AutoTokenizer` from HuggingFace Transformers. After a question is passed through BERT, the model outputs a sequence of embeddings, in which the [CLS] embedding is extracted. A linear layer is then applied to the [CLS] embedding, which then represents the logits for each answer choice. <br><br>

## Results Analysis
From our testing, the classification approach outperforms the generative approach. There are several reasons that could explain this difference:

**Option Evaluation**: One possible reason for this difference could be in the way both models evaluate the best option.
- In the classification approach (BERT model), each choice is independently evaluated and given a probability.
- In the generative approach (GPT-2 model), the model needs to keep track of all 4 options at once in order to generate a prediction. On top of learning how to map from the sequence to an answer choice, the model also needs to learn the way the options are structured.<br><br>

**Stability**: Another possible reason for this difference could be in the way both models deal with errors or misinterpretations.
- In the classification approach (BERT model), each choice is independently evaluated. If part of the question or fact is misinterpreted when evaluating one option, that error does not affect the evaluation of other options.
- In the generative approach (GPT-2 model), all answer choices are evaluated simultaneously. If the model misinterprets the question or fact, the model will evaluate all options incorrectly which can cause it to pick the incorrect answer.<br><br>