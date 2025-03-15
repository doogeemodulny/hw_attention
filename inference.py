import torch
from preprocessing import DEVICE, word_field
from model import EncoderDecoder


def summarize_text(model, text, word_field, max_length=100):
    text_field = [word_field.preprocess(text)]
    indexed_input = word_field.process(text_field).to(DEVICE)

    source_inputs = indexed_input.transpose(0, 1)
    source_mask = (source_inputs != word_field.vocab.stoi['<pad>']).unsqueeze(-2)
    summary_ids = model.generate_summary(source_inputs, source_mask, max_length=max_length)
    summary_text = ' '.join([word_field.vocab.itos[idx] for idx in summary_ids[0][0].cpu().numpy() if
                             idx not in [word_field.vocab.stoi['<pad>'], word_field.vocab.stoi['<s>'],
                                         word_field.vocab.stoi['</s>']]])

    return summary_text


texts = [
        "Госсекретаря США Марко Рубио встретили в аэропорту Канады, расстелив перед трапом самолёта красную ковровую дорожку, которая вела прямо в лужу.",
        "У президента России Владимира Путина «нет оправданий» для нападения на другую страну, но обсуждать интересы Москвы необходимо, считает бывший канцлер ФРГ Ангела Меркель",
        "В России этой весной пенсионеров ожидают три типа денежных доплат"]
model = EncoderDecoder(source_vocab_size=55917, target_vocab_size=55917).to(DEVICE)
model.load_state_dict(torch.load('trained_model_no_bert_emb.pt', map_location=torch.device('cpu')), strict=False)
model.eval()
for text in texts:
    summary = summarize_text(model, text, word_field, max_length=20)
    print("Суммари:", summary[6:].capitalize() + '.')
