import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW,SGD
from transformers import get_scheduler
import torch.nn as nn
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from utils import *
import os

# Set seed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Tokenize the input
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Core training function
def do_train(args, model, train_dataloader, save_dir="./out"):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model.train()
    progress_bar = tqdm(range(num_training_steps))

    ################################
    ##### YOUR CODE BEGINGS HERE ###

    loss_fun = nn.CrossEntropyLoss()


    for epoch in range(args.num_epochs):
        # correct = 0
        # total = 0
        # optimizer.zero_grad()
        # Reset the learning rate scheduler if required
        
        # i=0
        for outputs in train_dataloader:
            correct = 0
            total = 0
            # i+=1
            # if i==100:
            #     break
            input_ids, token_type_ids, attention_mask,label=outputs['input_ids'], outputs['token_type_ids'], outputs['attention_mask'], outputs['labels']
            input_ids, token_type_ids, attention_mask,label = (input_ids.to(device),
                                                                token_type_ids.to(device),
                                                                attention_mask.to(device),
                                                                label.to(device))
 
            # Forward pass
            outputs = model(input_ids=input_ids,token_type_ids=token_type_ids, 
                            attention_mask=attention_mask,labels=label)
            loss = outputs.loss
            # Compute loss
            # loss = loss_fun(pred.logits, label)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Calculate accuracy
            predicted = torch.argmax(outputs.logits, -1)
            total += len(label)
            correct += (predicted == label).sum().item()

            # Display metrics
            tqdm.write(f"Loss: {loss.item():.4f}, Accuracy: {correct / total:.4f}")

            # Update progress bar
            progress_bar.update(1)

        # print(f"Epoch {epoch + 1}/{args.num_epochs} - Training accuracy: {correct / total:.4f}")
        lr_scheduler.step()
    # Closing the progress bar after training
    progress_bar.close()
    # Implement the training loop --- make sure to use the optimizer and lr_sceduler (learning rate scheduler)
    # Remember that pytorch uses gradient accumumlation so you need to use zero_grad (https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html)
    # You can use progress_bar.update(1) to see the progress during training
    # You can refer to the pytorch tutorial covered in class for reference


    ##### YOUR CODE ENDS HERE ######

    print("Training completed...")
    print("Saving Model....")
    model.save_pretrained(save_dir)

    return


# Core evaluation function
def do_eval(eval_dataloader, output_dir, out_file):
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    model.eval()

    metric = evaluate.load("accuracy")
    out_file = open(out_file, "w")

    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

        # write to output file
        for pred, label in zip(predictions, batch["labels"]):
                out_file.write(f"{pred.item()}\n")
                out_file.write(f"{label.item()}\n")
    out_file.close()
    score = metric.compute()

    return score


# Created a dataladoer for the augmented training dataset
def create_augmented_dataloader(args, dataset):
    ################################
    ##### YOUR CODE BEGINGS HERE ###
    # Here, 'dataset' is the original dataset. You should return a dataloader called 'train_dataloader' -- this
    # dataloader will be for the original training split augmented with 5k random transformed examples from the training set.
    # You may find it helpful to see how the dataloader was created at other place in this code.
    def delete_random_word(sentence):
        words = sentence.split()
        if len(words) == 1:
            return sentence
        random_word = random.choice(words)
        new_words = [word for word in words if word != random_word]
        return ' '.join(new_words)
    
    sentence=dataset['train']['text']    

    # Select 5k random samples from the training set for augmentation
    num_augmented_samples = 5000
    if len(sentence) < num_augmented_samples:
        num_augmented_samples = len(sentence)

    indices = random.sample(range(len(sentence)), num_augmented_samples)
    selected_samples = [sentence[idx] for idx in indices]


    augmented_samples = [delete_random_word(sentence) for sentence in selected_samples]

    # Create a new dataset with both original and augmented samples
    original_samples = [{"text": sentence, "label": dataset['train']['label'][i]} for i, sentence in enumerate(sentence)]
    augmented_data = [{"text": sentence, "label": (dataset['train']['label'])} for sentence in augmented_samples]
    
    # Combine original and augmented samples
    combined_dataset = original_samples + augmented_data
    
    # Tokenize the combined dataset
    def tokenize_function(example):
        return tokenizer(example['text'], padding="max_length", truncation=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=args.batch_size,
        shuffle=True
    )


    ##### YOUR CODE ENDS HERE ######

    return train_dataloader 


# Create a dataloader for the transformed test set
def create_transformed_dataloader(args, dataset, debug_transformation):
    # Print 5 random transformed examples
    if debug_transformation:
        small_dataset = dataset["test"].shuffle(seed=42).select(range(5))
        small_transformed_dataset = small_dataset.map(custom_transform, load_from_cache_file=False)
        for k in range(5):
            print("Original Example ", str(k))
            print(small_dataset[k])
            print("\n")
            print("Transformed Example ", str(k))
            print(small_transformed_dataset[k])
            print('=' * 30)

        exit()

    transformed_dataset = dataset["test"].map(custom_transform, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_tokenized_dataset.remove_columns(["text"])
    transformed_tokenized_dataset = transformed_tokenized_dataset.rename_column("label", "labels")
    transformed_tokenized_dataset.set_format("torch")

    transformed_val_dataset = transformed_tokenized_dataset
    eval_dataloader = DataLoader(transformed_val_dataset, batch_size=args.batch_size)

    return eval_dataloader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--train", action="store_true",default=False, help="train a model on the training data")
    parser.add_argument("--train_augmented", action="store_true",default=False, help="train a model on the augmented training data")
    parser.add_argument("--eval", action="store_true",default=False, help="evaluate model on the test set")
    parser.add_argument("--eval_transformed", action="store_true",default=True, help="evaluate model on the transformed test set")
    parser.add_argument("--model_dir", type=str, default="./out_augmented/")
    parser.add_argument("--debug_train", action="store_true",default=False,
                        help="use a subset for training to debug your training loop")
    parser.add_argument("--debug_transformation", action="store_true",
                        help="print a few transformed examples for debugging")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    global device
    global tokenizer

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Tokenize the dataset
    dataset = load_dataset("imdb")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Prepare dataset for use by model
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(4000))
    small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

    # Create dataloaders for iterating over the dataset
    if args.debug_train:
        train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(small_eval_dataset, batch_size=args.batch_size)
        print(f"Debug training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")
    else:
        train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=args.batch_size)
        print(f"Actual training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")

    # Train model on the original training dataset
    if args.train:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model.to(device)
        do_train(args, model, train_dataloader, save_dir="./out")
        # Change eval dir
        args.model_dir = "./out"

    # Train model on the augmented training dataset
    if args.train_augmented:
        train_dataloader = create_augmented_dataloader(args,dataset)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model.to(device)
        do_train(args, model, train_dataloader, save_dir="./out_augmented")
        # Change eval dir
        args.model_dir = "./out_augmented"

    # Evaluate the trained model on the original test dataset
    if args.eval:
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        out_file = out_file + "_original.txt"
        score = do_eval(eval_dataloader, args.model_dir, out_file)
        print("Score: ", score)

    # Evaluate the trained model on the transformed test dataset
    if args.eval_transformed:
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        out_file = out_file + "_transformed.txt"
        eval_transformed_dataloader = create_transformed_dataloader(args, dataset, args.debug_transformation)
        score = do_eval(eval_transformed_dataloader, args.model_dir, out_file)
        print("Score: ", score)
