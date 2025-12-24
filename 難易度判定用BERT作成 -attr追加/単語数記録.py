import spacy
import json
import nltk
from datasets import load_dataset
import Spacy_hantei_Mk5 as Mk4
import Discode通知 as disco
# from collections import Counter # 今回は自作ロジックを使うのでコメントアウト

nlp = spacy.load("en_core_web_trf")
word_list = {}
sentencelist = {}
wiki_article_num = 3000

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab') # 環境によってはこれも必要な場合があります

print("Wikipediaデータセットをロードしています...")
wiki_dataset = load_dataset("wikimedia/wikipedia", name="20231101.en", split='train', streaming=True)
print("ロードが完了しました。")

disco.main(f"文章難易度データセット作成用、データセット構築を開始します。\n")

def Word_Num_Maker(sentence):
    # print() # 大量に空行が出ると邪魔になるのでコメントアウトしました
    doc = nlp(sentence)
    sentence_word_list = [token.text for token in doc]

    for word in sentence_word_list:
        # 【修正】高速化: list()変換をやめて直接辞書をチェック
        # 【修正】ロジック: あるなら足す、ないなら1を入れる
        if word in word_list:
            word_list[word] += 1
        else:
            word_list[word] = 1

def Make_sentece_difference(sentence):
    results = Mk4.main(sentence)
    for result in results:
        if result in sentencelist:
            sentencelist[result] += 1
        else :
            sentencelist[result] = 1



    
def Make_list_manyThings():
    j = 0 # 【修正】初期化を追加
    
    for article in wiki_dataset:
        j += 1
        if(j%600 == 0):
            disco.main(f"現在の処理件数は{j}件です。\n")

        if j > wiki_article_num: # >= だと指定数より1つ少なくなる場合があるので > に調整（お好みで）
            break
        
        print(f"Processing article {j}...") # 進捗表示
        
        article_text = article['text']
        sentences = nltk.sent_tokenize(article_text)
        for sentence in sentences:
            Word_Num_Maker(sentence)
            Make_sentece_difference(sentence)
    
    # ゼロ除算回避（念のため）
    if not word_list or not sentencelist:
        return {},{}

    total_count = sum(word_list.values())
    total_count2 = sum(sentencelist.values())
    
    # 【修正】辞書をコピーして作成（元のカウントデータを破壊しないため）
    percent_of_words = word_list.copy()
    percent_of_sentences = sentencelist.copy()
    
    for key in word_list.keys():
        percent_of_words[key] = word_list[key] / total_count
    for key in sentencelist.keys():
        percent_of_sentences[key] = sentencelist[key] / total_count2



    return percent_of_words,percent_of_sentences

if __name__ == "__main__":
    result1,result2 = Make_list_manyThings()
    # 全部は多いので、試しに一部を表示するか、長さを確認
    print(f"Total unique words: {len(result1)}")

    with open("word_probs.json", "w", encoding="utf-8") as f:
        json.dump(result1, f, indent=4, ensure_ascii=False)
    
    # 2. 文法確率の保存 (result2)
    with open("grammar_probs.json", "w", encoding="utf-8") as f:
        json.dump(result2, f, indent=4, ensure_ascii=False)

    print("jsonファイルへの保存が完了しました。")
    disco.main(f"難易度判定用の学習データの作成(第一段階 : 統計調査)が終了しました。\n 第二段階の本学習を開始してください。\nTotal unique words: {len(result1)}\n")

