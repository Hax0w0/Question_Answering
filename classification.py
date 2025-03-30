# Class: Northwestern CS 461 Winter 2025
# ---------------------------------------

# Professor: David Demeter
# ---------------------------------------

# Contributers:
# ---------------------------------------
#   Raymond Gu
#   Maanvi Sarwadi

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, BertModel

def main():

    # Set up the random seed
    torch.manual_seed(0)

    # Check if we can use the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Get the training, validation, and test datasets
    train_formatted = format_data("train_complete.jsonl")
    train_dataset = create_dataset(train_formatted, tokenizer)

    valid_formatted = format_data("valid_complete.jsonl")
    valid_dataset = create_dataset(valid_formatted, tokenizer)

    test_formatted = format_data("test_complete.jsonl")
    test_dataset = create_dataset(test_formatted, tokenizer)

    # Set up the model
    model = ClassificationApproach().to(device)

    # Set up the hyperparameters
    batch_size = 16
    learning_rate = 5e-6
    num_epochs = 10
    loadname = None

    # Get zero-shot accuracy for validation and test sets
    test_zero_shot = get_accuracy(model, test_dataset, batch_size, device)
    valid_zero_shot = get_accuracy(model, valid_dataset, batch_size, device)

    # Load the weights if provided, otherwise train the model
    if loadname is not None:
        print("Loading pretrained weights...\n")
        model.load_state_dict(torch.load(loadname))
    else:
        train(model, train_dataset, valid_dataset, batch_size, num_epochs, learning_rate, device)
        model.load_state_dict(torch.load('classification_model_weights.pth'))

    # Get final accuracy on the validation and test sets
    test_final_accuracy = get_accuracy(model, test_dataset, batch_size, device)
    valid_final_accuracy = get_accuracy(model, valid_dataset, batch_size, device)

    # Print the results
    print("Final Results:")
    print("------------------------------------------")
    print("   Test Zero Shot Accuracy: ", test_zero_shot, "%")
    print("   Test Final Accuracy: ", test_final_accuracy, "%")

    print("")
    print("   Validation Zero Shot Accuracy: ", valid_zero_shot, "%")
    print("   Validation Final Accuracy: ", valid_final_accuracy, "%")

def create_dataset(formatted_data, tokenizer):
    """
    Description:
        Creates the dataset for training, validation, or test data.

    Inputs:
        - Formatted_Data: A list with each of the questions.
        - Tokenizer: The tokenizer that is used to tokenize each sequence.

    Output:
        - The TensorDataset for the data.
    """
    
    # Intialize variable to return
    text_list = []
    attention_mask_list = []
    labels_list = []

    for question in formatted_data:

        # Get each part of the question
        texts = [option[0] for option in question]
        labels = [option[1] for option in question]
        correct_index = labels.index(1)

        # Use the tokenizer to tokenize the text and get the attention mask
        output = tokenizer(texts, padding="max_length", max_length=100, truncation=False, return_tensors="pt")
        tokenized_texts = output["input_ids"]
        attention_mask = output["attention_mask"]

        text_list.append(tokenized_texts)
        attention_mask_list.append(attention_mask)
        labels_list.append(correct_index)

    # Convert the lists into tensors
    text_tensor = torch.stack(text_list)
    attention_mask_tensor = torch.stack(attention_mask_list)
    labels_tensor = torch.tensor(labels_list)

    # Put the tensors into a TensorDataset
    dataset = TensorDataset(text_tensor, attention_mask_tensor, labels_tensor)
    
    return dataset

def format_data(file_name):
    """
    Description:
        This function formats each question in the way shown below:
        Question = [Option_A, Option_B, Option_C, Option_D]

        Each option is formatted as shown below (using Option_A as an example):
        Option_A = [Text_A, Correctness_A]

        The text for each option is formmatted as shown below (using Text_A as an example):
        Text_A = [CLS] {Fact} [SEP] {Question} [SEP] {Answer_Choice A} [SEP]

    Inputs:
        - File_Name: The name of the file that we're trying to parse.

    Output:
        - Formatted_Data: A list with each question formatted.
    """

    formatted_data = []
    answers = ['A','B','C','D']

    with open(file_name) as json_file:
        json_list = list(json_file)

    for i in range(len(json_list)):

        # Get the example in the training set.
        json_str = json_list[i]
        result = json.loads(json_str)
        
        # Get the question and the fact used to answer it.
        question = result['question']['stem']
        fact = result['fact1']

        # Get the index of the answer (A=0, B=1, C=2, D=3)
        ans = answers.index(result['answerKey'])
        
        formatted_question = []
        for j in range(4):

            # Format each instance
            answer_choice = result['question']['choices'][j]['text']
            text = "[CLS] " + fact + " [SEP] " + question + " [SEP] " + answer_choice + " [SEP]"

            # Get the label for the choice (Correct = 1, Incorrect = 0)
            if j == ans: label = 1
            else: label = 0

            formatted_question.append([text,label])

        formatted_data.append(formatted_question)

    return formatted_data

class ClassificationApproach(nn.Module):
    def __init__(self):

        super(ClassificationApproach, self).__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):

        batch_size, num_choices, seq_length = input_ids.size()
        flat_input_ids = input_ids.view(-1, seq_length)
        flat_attention_mask = attention_mask.view(-1, seq_length)

        outputs = self.model(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        logits = self.linear(cls_embeddings)
        logits = logits.view(batch_size, num_choices)

        return logits

def train(model, train_dataset, valid_dataset, batch_size, num_epochs, learning_rate, device):

    # Set the model to training mode and initialize the optimizer & criterion
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Create batches for the training dataset
    train_batches = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Create a list to store the validation accuracy across epochs
    valid_acc = []

    for epoch in range(num_epochs):

        for batch in train_batches:

            tokens, attention_mask, labels = batch
            tokens = tokens.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(tokens, attention_mask)

            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        
        valid_accuracy = get_accuracy(model, valid_dataset, batch_size, device)
        valid_acc.append(valid_accuracy)
        model.train()

        # Stop early if performance on the validation set doesn't increase
        if (epoch > 0 and valid_acc[-2] > valid_acc[-1]):
            break

        # Save the weights if the model improved
        torch.save(model.state_dict(), 'classification_model_weights.pth')

def get_accuracy(model, dataset, batch_size, device):

    # Set the model to evaluation mode
    model.eval()

    # Create batches
    batches = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Create variables to keep track of accuracy
    total_correct = 0
    total_predictions = 0

    with torch.no_grad():

        for batch in batches:

            tokens, attention_mask, labels = batch
            tokens = tokens.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(tokens, attention_mask)

            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = (total_correct / total_predictions) * 100
    return accuracy

if __name__ == "__main__":
    main()
