import torch

from preprocessing import DEVICE, word_field, BucketIterator, train_dataset
from model import EncoderDecoder
from visualize_att import AttentionVisualizer

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
        "В России этой весной пенсионеров ожидают три типа денежных доплат", 
        "Глава МИД Ирана рассказал о готовности страны к ведению конфликта",
        "Медведев в пух и прах разнес Зеленского за выдуманный разговор с Макроном: «Устроил настоящий цирк»",
        "Папа Римский впервые появился перед верующими после госпитализации из-за болезни",
        "Мэр Стамбула Экрем Имамоглу арестован по решению суда, который удовлетворил ходатайство прокуратуры об избрании меры пресечения на фоне обвинений в коррупции и пособничестве терроризму.",
        "Китайские ученые предложили превратить Тибетское нагорье в новый сельскохозяйственный регион из-за ускоряющихся изменений климата, которые могут повлиять на традиционные сельхозрайоны.",
        "Президент Чехии хотел поддержать Киев, но случайно вскрыл гнойник Запада: многолетнее лицемерие и ложь",
        "Подростков задержали на Ставрополье за поджог здания Союза ветеранов, который они совершили по указанию кураторов из интернета."]
model = EncoderDecoder(source_vocab_size=53879, target_vocab_size=53879).to(DEVICE)
model.load_state_dict(torch.load('trained_model_bert_emb_50_01_256_1024.pt', map_location=torch.device('cpu')), strict=False)
model.eval()
for text in texts:
    summary = summarize_text(model, text, word_field, max_length=20)
    print("Суммари:", summary[6:].capitalize() + '.')


# train_iter = BucketIterator(train_dataset, batch_size=1, device=DEVICE)
# visualizer = AttentionVisualizer(model, word_field)
# for ex_i, ex_v in enumerate(train_iter):
#     if ex_i==3:
#         break
#     visualizer.attention_mech_visualization(ex_v, ex_i)





