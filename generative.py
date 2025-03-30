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
from torch import tensor
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():

    # Set up the random seed
    torch.manual_seed(0)

    # Check if we can use the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({"pad_token": "[PAD]",
                                  "additional_special_tokens": ["[START]", "[SEP]", "[ANSWER]"]})
    tokenizer.pad_token = tokenizer.eos_token

    # Intialize the GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    # Get the training, validation, and test datasets
    train_dataset, valid_dataset, test_dataset = get_datasets(tokenizer)

    # Set up the hyperparameters
    batch_size = 1
    learning_rate = 3e-5
    num_epochs = 10
    loadname = None

    # Get zero-shot accuracy for validation and test sets
    test_zero_shot = get_accuracy(model, tokenizer, test_dataset, batch_size, device)
    valid_zero_shot = get_accuracy(model, tokenizer, valid_dataset, batch_size, device)

    # Load the weights if provided, otherwise train the model
    if loadname is not None:
        print("Loading pretrained weights...\n")
        model.load_state_dict(torch.load(loadname))
    else:
        train(model, train_dataset, valid_dataset, tokenizer, batch_size, num_epochs, learning_rate, device)
        model.load_state_dict(torch.load('generative_model_weights.pth'))
    
    # Get final accuracy on the validtaion and test sets
    test_final_accuracy = get_accuracy(model, tokenizer, test_dataset, batch_size, device)
    valid_final_accuracy = get_accuracy(model, tokenizer, valid_dataset, batch_size, device)

    # Print final results
    print("Final Results:")
    print("------------------------------------------")
    print("   Test Zero Shot Accuracy: ", test_zero_shot, "%")
    print("   Test Final Accuracy: ", test_final_accuracy, "%")

    print("")
    print("   Validation Zero Shot Accuracy: ", valid_zero_shot, "%")
    print("   Validation Final Accuracy: ", valid_final_accuracy, "%")

def get_datasets(tokenizer):
    """
    Description:
        This function gets the training, validation, and test datasets.

    Input:
        - Tokenizer: The tokenizer that is used to tokenize each sequence.

    Output:
        - Train_Dataset: The training dataset.
        - Valid_Dataset: The validation dataset.
        - Test_Dataset: The test dataset.
    """

    # Parse through the training, validation, and test data
    train, train_answers = format_data("train_complete.jsonl", append_answer=True)
    valid, valid_answers = format_data("valid_complete.jsonl", append_answer=False)
    test, test_answers = format_data("test_complete.jsonl", append_answer=False)

    # Tokenize the training, validation, and test data
    train_tokenized = tokenize_data(train, tokenizer)
    valid_tokenized = tokenize_data(valid, tokenizer)
    test_tokenized = tokenize_data(test, tokenizer)

    # Convert answer lists to PyTorch tensors
    train_answers = tensor(train_answers, dtype=torch.long)
    valid_answers = tensor(valid_answers, dtype=torch.long)
    test_answers = tensor(test_answers, dtype=torch.long)

    # Organize the data into a tensor dataset
    train_dataset = TensorDataset(train_tokenized["input_ids"], train_tokenized["attention_mask"], train_answers)
    valid_dataset = TensorDataset(valid_tokenized["input_ids"], valid_tokenized["attention_mask"], valid_answers)
    test_dataset = TensorDataset(test_tokenized["input_ids"], test_tokenized["attention_mask"], test_answers)

    return train_dataset, valid_dataset, test_dataset

def format_data(file_name, append_answer):
    """
    Description:
        This function formats each question in the TRAINING data in the way shown below:
            [START] {Fact} {Stem} [A] {Option A} [B] {Option B} [C] {Option C} [D] {Option D} [ANSWER] {Correct Label}

        This function formats each question in the VALIDATION and TEST data in way the shown below:
            [START] {Fact} {Stem} [A] {Option A} [B] {Option B} [C] {Option C} [D] {Option D} [ANSWER] {Correct Label}

    Inputs:
        - File_Name: The name of the file that we're trying to parse.
        - Append_Answer: This boolean argument tells us if we should append the answer or not (answer should be appended for training data).

    Output:
        - Formatted_Data: A list with each question formatted.
        - Answers: A list of the answers for each question.
    """
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3} 

    formatted_data = []
    answers = []

    with open(file_name) as json_file:
        json_list = list(json_file)
        num_questions = len(json_list)

    for i in range(num_questions):

        # Get the example in the training set.
        json_str = json_list[i]
        result = json.loads(json_str)
        
        # Get all relevant parts of the question.
        stem = result['question']['stem']
        fact = result['fact1']
        a = result['question']['choices'][0]['text']
        b = result['question']['choices'][1]['text']
        c = result['question']['choices'][2]['text']
        d = result['question']['choices'][3]['text']
        answer = result['answerKey']
        
        formatted_question = "[START] " + fact + " [SEP] " + stem + " [A] " + a + " [B] " + b + " [C] " + c + " [D] " + d + " [ANSWER] "

        if (append_answer):
            formatted_question = formatted_question + answer

        formatted_data.append(formatted_question)
        answers.append(answer_map.get(answer, -1))

    return formatted_data, answers

def tokenize_data(formatted_data, tokenizer):
    """
    Description:
        This function takes the formatted data and tokenizes each sequence (while also adding padding).

    Inputs:
        - Formatted_Data: A list with each of the questions.
        - Tokenizer: The tokenizer that is used to tokenize each sequence.

    Output:
        - Tokenized_Data: A dictionary that contains the tokenized version of each question and the attention mask.
    """
    tokenized_data = tokenizer(formatted_data, padding="longest", return_tensors="pt")
    return tokenized_data

def train(model, train_dataset, valid_dataset, tokenizer, batch_size, num_epochs, learning_rate, device):

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Use DataLoader to create batches
    train_batches = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Create a list to store the validation accuracy across epochs
    valid_acc = []

    model.train()

    for epoch in range(num_epochs):

        for batch in train_batches:

            tokens, attention_mask, answers = batch
            tokens = tokens.to(device)
            attention_mask = attention_mask.to(device)
            answers = answers.to(device)
            
            labels = tokens.clone()
            labels[:, :-1] = -100

            # Put the input through the GPT-2 model
            outputs = model(tokens, attention_mask=attention_mask, labels=labels)
            
            # Calculate the loss and the gradient
            loss = outputs.loss
            loss.backward()

            # Update the model and then zero out the gradient
            optimizer.step()
            optimizer.zero_grad()

        # Get the accuracy of the model on the validation set
        valid_accuracy = get_accuracy(model, tokenizer, valid_dataset, batch_size, device)
        valid_acc.append(valid_accuracy)
        model.train()

        # Stop early if performance on the validation set doesn't increase
        if (epoch > 0 and valid_acc[-2] > valid_acc[-1]):
            break

        # Save the weights if the model improved
        torch.save(model.state_dict(), 'generative_model_weights.pth')
        
def get_accuracy(model, tokenizer, dataset, batch_size, device):

    # Set the model to evaluation model
    model.eval()

    # Use DataLoader to create batches
    batches = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Set up variables to track accuracy
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in batches:

            tokens, attention_mask, answers = batch
            tokens = tokens.to(device)
            attention_mask = attention_mask.to(device)
            answers = answers.to(device)

            outputs = model.generate(tokens, 
                                     attention_mask=attention_mask,
                                     num_beams=5,
                                     min_new_tokens=1,
                                     max_new_tokens=1,
                                     pad_token_id=tokenizer.pad_token_id)

            # Loop through all the outputs generated for each question
            for i in range(len(outputs)):

                # Get the prediction for each question
                generated_text = tokenizer.decode(outputs[i], skip_special_tokens=False)
                predicted_answer = generated_text.split("[ANSWER]")[-1].strip()

                # Get the correct answer for each question
                correct_answer = chr(answers[i].item() + 65)

                if predicted_answer == correct_answer:
                    correct_predictions += 1
                total_predictions += 1

        accuracy = (correct_predictions / total_predictions) * 100
        return accuracy

if __name__ == "__main__":
    main()
