import torch.optim as optim
from tqdm.auto import tqdm
from rouge import Rouge
from preprocessing import train_iter, test_iter
from model import *


""" Optimizer """


class NoamOpt(object):
    def __init__(self, model_size, factor=2, warmup=4000, optimizer=None):
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


""" ROUGE metric """


def evaluate_rouge(model, val_iter):
    model.eval()
    rouge = Rouge()
    references = []
    hypotheses = []

    with torch.no_grad():
        with tqdm(total=len(val_iter)) as progress_bar:
            for batch in val_iter:
                source_inputs, target_inputs, source_mask, target_mask = convert_batch(batch)

                summaries = model.generate_summary(source_inputs, source_mask)

                for tgt in target_inputs[:, 1:]:
                    references.append(' '.join(map(str, tgt.tolist())).strip())
                for summary in summaries[0]:
                    generated_summary = ' '.join(map(str, summary.tolist())).strip()
                    hypotheses.append(generated_summary)
                progress_bar.update()

    scores = rouge.get_scores(references, hypotheses, avg=True)
    return scores


""" Training """
tqdm.get_lock().locks = []
train_losses_history = []
val_losses_history = []


def do_epoch(model, criterion, data_iter, optimizer=None, name=None):
    epoch_loss = 0

    is_train = not optimizer is None
    name = name or ''
    model.train(is_train)

    batches_count = len(data_iter)

    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count) as progress_bar:
            for i, batch in enumerate(data_iter):
                source_inputs, target_inputs, source_mask, target_mask = convert_batch(batch)
                logits = model.forward(source_inputs, target_inputs[:, :-1], source_mask, target_mask[:, :-1, :-1])

                logits = logits.contiguous().view(-1, logits.shape[-1])
                target = target_inputs[:, 1:].contiguous().view(-1)
                loss = criterion(logits, target)

                epoch_loss += loss.item()

                if optimizer:
                    optimizer.optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                progress_bar.update()
                progress_bar.set_description('{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(name, loss.item(),
                                                                                         math.exp(loss.item())))

            progress_bar.set_description('{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(
                name, epoch_loss / batches_count, math.exp(epoch_loss / batches_count))
            )
            progress_bar.refresh()

    return epoch_loss / batches_count


def fit(model, criterion, optimizer, train_iter, epochs_count=1, val_iter=None):
    best_val_loss = None
    best_model = None
    name_prefix = ""
    for epoch in range(epochs_count):
        name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)
        train_loss = do_epoch(model, criterion, train_iter, optimizer, name_prefix + 'Train:')
        train_losses_history.append(train_loss)
        if val_iter:
            val_loss = do_epoch(model, criterion, val_iter, None, name_prefix + '  Val:')
            val_losses_history.append(val_loss)
            if best_val_loss is None or val_loss < best_val_loss:
                best_model = model
                best_val_loss = val_loss
    rouge_scores = evaluate_rouge(model, val_iter)
    print(
        f"{name_prefix} ROUGE: F1 = {rouge_scores['rouge-l']['f']:.4f}, Precision = {rouge_scores['rouge-l']['p']:.4f}, Recall = {rouge_scores['rouge-l']['r']:.4f}")
    return best_model


""" Fit model with LabelSmoothing loss and extended hidden size """
if __name__ == "main":
    model = EncoderDecoder(source_vocab_size=len(word_field.vocab),
                           target_vocab_size=len(word_field.vocab), d_model=400, d_ff=1600).to(DEVICE)
    pad_idx = word_field.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1).to(DEVICE)
    optimizer = NoamOpt(model.d_model)
    model = fit(model, criterion, optimizer, train_iter, epochs_count=40, val_iter=test_iter)

    torch.save(model.state_dict(), "trained_model_no_bert_emb_1.pt")
