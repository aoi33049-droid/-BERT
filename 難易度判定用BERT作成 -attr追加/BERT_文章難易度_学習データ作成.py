import spacy
import json
import nltk
from datasets import load_dataset
import Spacy_hantei_Mk5 as Mk4
import math
import Discode通知 as disco
# from collections import Counter # 今回は自作ロジックを使うのでコメントアウト

nlp = spacy.load("en_core_web_trf")
word_list = {}
sentencelist = {}
final_datasets = {}
wiki_article_num = 3000

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab') # 環境によってはこれも必要な場合があります

print("Wikipediaデータセットをロードしています...")
wiki_dataset = load_dataset("wikimedia/wikipedia", name="20231101.en", split='train', streaming=True)
print("ロードが完了しました。")
disco.main("難易度データセット作成を開始します")



def make_difference_sentence():
    with open("word_probs.json", "r", encoding="utf-8") as f:
        word_data = json.load(f)
    
    with open("grammar_probs.json", "r", encoding="utf-8") as f:
        grammer_data = json.load(f)
    
    j = 0 # 【修正】初期化を追加
    
    alpha = 1
    beta = 1
    theta = 1

    for article in wiki_dataset:
        j += 1
        if(j%600 == 0):
            disco.main(f"現在の学習件数は{j}件です")
        if j > wiki_article_num: # >= だと指定数より1つ少なくなる場合があるので > に調整（お好みで）
            break
        
        print(f"Processing article {j}...") # 進捗表示
        
        article_text = article['text']
        sentences = nltk.sent_tokenize(article_text)
        for sentence in sentences:
            grammer_defference = 0
            word_defference = 0
            length_defference = 0
            total_defference = 0

            return_grammer = Mk4.main(sentence)
            for grammer in return_grammer:
                if grammer in grammer_data:
                    grammer_defference += -math.log(max(grammer_data[grammer],1e-10))
                else : grammer_defference += -math.log(1e-10)
            
            doc = nlp(sentence)
            sentence_word_list = [token.text for token in doc]

            wordcount = 0

            for word in sentence_word_list:
                wordcount += 1
                
                if word in word_data:
                    word_defference += -math.log(max(word_data[word],1e-10))
                else : word_defference += -math.log(1e-10)

            if wordcount != 0:
                word_defference = word_defference / wordcount
                length_defference = math.log(max(wordcount,1))
            else : 
                word_defference = 0
                length_defference = 0

            
            

            total_defference = alpha*grammer_defference + beta*word_defference + theta*length_defference
            final_datasets[sentence] = total_defference

def main():
    make_difference_sentence()
    with open("BERT_defference_datasets.json", "w", encoding="utf-8") as f:
        json.dump(final_datasets, f, indent=4, ensure_ascii=False)
    
    disco.main("学習を完了しました。BERTの学習を開始してください")


if __name__ == "__main__":
    main()