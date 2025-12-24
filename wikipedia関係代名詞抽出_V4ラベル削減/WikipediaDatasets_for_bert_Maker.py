import nltk
from datasets import load_dataset
import json
import time
import Spacy_hantei_Mk6 as Mk4 # 実行環境に合わせてコメントアウト解除
from collections import Counter
import numpy as np
from DB_hantei_kakunou import PatternRegistry
import random
import spacy
from transformers import pipeline
import Discode as disco

# ------------------------------------------------------------------

# 各種変数
wiki_article_num = 3000 # テスト用。本番ではもっと増やす
neg_ratio = 1 #なしラベル比率
output_file = "BERTdataset_準備.json"
stats_file = "BERTdataset_counter.json"
final_output_file = "BERTdatasets.json"
print("spacyをダウンロードします")
nlp = spacy.load("en_core_web_trf")
print("関係副詞省略判定用BERTをダウンロードします")
print("Loading bert-large-uncased... (これには数秒〜数分かかります)")

# "bert-large-uncased" を指定
# 理由: baseモデルよりも層が厚く、複雑な文脈（動詞の他動性や先行詞のニュアンス）を正確に捉えます。
# GPUがある場合は device=0 を指定すると爆速になります (CPUなら device=-1)
unmasker = pipeline('fill-mask', model='bert-large-uncased', device=-1)

dataset = []
label_counter = Counter()

# NLTKのデータダウンロード（初回のみ必要）
print("nltkをダウンロードします")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

print("Wikipediaデータセットをロードしています...")
# ネットワーク環境によっては時間がかかります
wiki_dataset = load_dataset("wikimedia/wikipedia", name="20231101.en", split='train', streaming=True)
print("すべてのロードが完了しました。")

print("bert用のデータセットを作成します")
start_time = time.time()

disco.main("BERTの学習を開始します。文法判定モデル\n")

def hantei(sentence: str):
    # Mk4.mainが返すリストを受け取る
    Serch_num = Mk4.main(sentence,nlp,unmasker)
    if Serch_num:
        return Serch_num
    else :
        print("Error")
        exit(1)

def expand_dataset_for_bert(intermediate_file, db_file, final_output_file):
    print(f"BERT用データ拡張を開始: {intermediate_file} -> {final_output_file}")
    
    with open(intermediate_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    with open(db_file, 'r', encoding='utf-8') as f:
        label_db = json.load(f)
    
    # DB(統計ファイル)のキーをすべて数値化してリストにする
    all_ids = [int(k) for k in label_db.keys()]
    
    # IDの最大値を取得
    max_id = max(all_ids) if all_ids else 0
    
    # 【修正1】最大IDが4なら、0~4の5個の要素が必要なので +1 する
    num_classes = max_id + 1

    print(f"最大ID: {max_id}, ベクトル次元数: {num_classes}")

    expanded_dataset = []

    for entry in raw_data:
        text = entry["text"]
        label_ids = entry["label_ids"]

        # Multi-Hotベクトルの作成
        labels_vec = [0] * num_classes
        
        for lid in label_ids:
            if lid < num_classes:
                labels_vec[lid] = 1
            else:
                # 理論上ここには来ないはずだが、安全策
                print(f"警告: 想定外のID {lid} (max: {max_id})")

        expanded_dataset.append({
            "text": text,
            "label_ids": label_ids,
            "labels_vec": labels_vec
        })

    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(expanded_dataset, f, ensure_ascii=False, indent=2)

    print(f"拡張完了: {final_output_file} を作成しました。")

    end_time = time.time()

    Result_Time = end_time - start_time
    minutes, seconds = divmod(Result_Time, 60)
    print(f"処理を完了しました。実行時間 : {minutes:.0f}分{seconds:.1f}秒")

def balance_dataset(input_file, output_file, neg_ratio=2.0):
    """
    正解データ数に基づいて、なしデータを間引く関数
    neg_ratio: 正解データ1件につき、なしデータを何件残すか（例: 2.0なら 1:2）
    """
    print("データのバランス調整（アンダーサンプリング）を開始...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    positives = [] # ラベルがあるデータ
    negatives = [] # ラベルがないデータ

    # 1. データを分類
    registry = PatternRegistry()
    for entry in raw_data:
        Nothing_id = registry.get_id("なし")
        if entry["label_ids"][0] != Nothing_id:
            positives.append(entry)
        else:
            negatives.append(entry)

    n_pos = len(positives)
    n_neg_original = len(negatives)
    
    print(f"  - ラベルあり: {n_pos} 件")
    print(f"  - ラベルなし(元): {n_neg_original} 件")

    # 2. 残す「なしデータ」の数を計算
    # 正解数 × 比率 （ただし、元の数より多くならないようにする）
    n_keep = int(n_pos * neg_ratio)
    n_keep = min(n_keep, n_neg_original)

    # 3. ランダムに抽出
    if n_pos == 0:
        print("警告: ラベルありデータが0件です。調整をスキップします。")
        sampled_negatives = negatives # あるいは []
    else:
        sampled_negatives = random.sample(negatives, n_keep)

    print(f"  - ラベルなし(調整後): {len(sampled_negatives)} 件 (比率 1:{neg_ratio})")

    # 4. 結合してシャッフル（学習順序の偏りを防ぐため）
    balanced_data = positives + sampled_negatives
    random.shuffle(balanced_data)

    # 5. 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=2)
        
    print(f"完了: {output_file} に保存しました。（合計 {len(balanced_data)} 件）")

    disco.main(f"完了: {output_file} に保存しました。（合計 {len(balanced_data)} 件）\n  - ラベルあり: {n_pos} 件\n - ラベルなし(元): {n_neg_original} 件 \n - ラベルなし(調整後): {len(sampled_negatives)} 件 (比率 1:{neg_ratio})")


def main():
    # streaming=Trueにした場合、rangeでのselectはできないため take を使います
    # または通常のload_datasetなら元の書き方でもOKです
    # ここでは汎用的に enumerate で回数制限をかけます
    
    processed_count = 0
    
    print("解析を開始します...")
    j = 0
    
    for article in wiki_dataset:
        j+=1
        if processed_count >= wiki_article_num:
            break
    
        article_text = article['text']
        sentences = nltk.sent_tokenize(article_text)
        if(j%600 == 0):
            disco.main(f"現在{j}番目の記事を解析しています。\n")

        for sentence in sentences:
            # 空の文や短すぎる文を除外するとノイズが減ります（任意）
            if not sentence.strip():
                continue

            results = hantei(sentence)
            print(f"現在{j}番目の記事を処理しています")

            data_entry = {
                "text": sentence,
                "label_ids": results
            }

            dataset.append(data_entry)
            label_counter.update(results)
        
        processed_count += 1
        print(f"記事 {processed_count}/{wiki_article_num} 完了")

    # 【修正2】ループの外で一括保存（高速化）
    print("中間データを保存中...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    balance_dataset(output_file, output_file, neg_ratio)


    print("統計データを保存中...")
    sorted_stats = dict(sorted(label_counter.items()))
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_stats, f, ensure_ascii=False, indent=2)

    # データ作成後に拡張処理を実行
    # 【修正3】文末のコロン削除
    expand_dataset_for_bert(output_file, stats_file, final_output_file)

if __name__ == "__main__":
    main()