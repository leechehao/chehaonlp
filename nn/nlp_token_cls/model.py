import time
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from transformers import get_linear_schedule_with_warmup, AdamW


class BertTokenCls(nn.Module):
    def __init__(self, bert_model_type, num_labels):
        super(BertTokenCls, self).__init__()
        config = AutoConfig.from_pretrained(bert_model_type)
        config.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_type)
        self.model = AutoModelForTokenClassification.from_pretrained(bert_model_type, config=config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        if labels is not None:
            loss, logits = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return loss, logits

        logits = self.model(input_ids=input_ids)[0] # (logits,) logits => [num_sample, len_subwords, num_labels]
        return logits

def setting(model, learning_rate, FULL_FINETUNING, total_steps, warmup_steps):
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=1e-8
    )

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=warmup_steps
    )

    return optimizer, scheduler

def train(model, dataloader, optimizer, scheduler, loss_values, device, log_file, FLAGS):
    # Put the model into training mode.
    model.train()

    # Reset the train loss for this epoch.
    train_loss = 0

    start_time = time.time()
    # Training loop
    for step, batch in enumerate(dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()

        # forward pass
        # This will return the loss because we have provided the `labels`.
        loss, logits = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        
        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # track train loss
        train_loss += loss.item()

        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=FLAGS['max_grad_norm'])

        # update parameters
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    duration = time.time() - start_time

    # Calculate the average loss over the training data.
    avg_train_loss = train_loss / len(dataloader)

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)
    print('Average train loss: {:.4f} ({:.3f} sec)'.format(avg_train_loss, duration))
    log_file.write('Average train loss: {:.4f} ({:.3f} sec)\n'.format(avg_train_loss, duration))

    return loss_values

def evaluate(model, dataloader, loss_values, device, log_file=None):
    predictions , true_labels = [], []

    # Put the model into evaluation mode
    model.eval()

    # Reset the validation loss for this epoch.
    eval_loss = 0

    start_time = time.time()
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions and loss.
            loss, logits = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()

        # track valid loss
        eval_loss += loss.item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)]) # [batch_size, max_len]
        true_labels.extend(label_ids)

    duration = time.time() - start_time

    avg_eval_loss = eval_loss / len(dataloader)
    loss_values.append(avg_eval_loss)
    print('Average Eval loss: {:.4f} ({:.3f} sec)'.format(avg_eval_loss, duration))
    if log_file is not None:
        log_file.write('Average Eval loss: {:.4f} ({:.3f} sec)\n'.format(avg_eval_loss, duration))

    return predictions, true_labels, loss_values
