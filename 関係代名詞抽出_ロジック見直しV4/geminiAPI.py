
import requests
import google.generativeai as genai
from PIL import Image
import json
import os
import sys
import glob
import re
import time

#api2 = AIzaSyDCSTXSLAqhhSVYPITwPov-rL3Mq2tDm1w




def gemini(num_sentences, bunnpou, file_name):
    """
    指定された文法ルールに基づき、指定された数の英文を生成してJSONファイルに保存します。
    """
    try:
        # --- API呼び出し部分 ---
        genai.configure(api_key="AIzaSyDCSTXSLAqhhSVYPITwPov-rL3Mq2tDm1w") # あなたのAPIキーに置き換えてください

        # ★修正点1: 正しいモデル名に変更
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""
        「{bunnpou}」のルールを使った英文を{num_sentences}個、コロン区切りで出力してください。
        最初と最後を波括弧{{}}で囲み、それ以外の説明や余計な文字は一切含めないでください。

        --------(例)---------
        {{If I were a bird, I would fly to you.:If she had studied harder, she would have passed the exam.:I wish I could speak English.}}
        ---------------------
        """
        response = model.generate_content(prompt)

        # --- テキスト加工部分 ---
        raw_text = response.text
        cleaned_text = raw_text.strip().strip('{}')
        sentences = cleaned_text.split(':')
        
        # APIから取得した新しいデータ
        update_data = {bunnpou: sentences}

        # --- ファイル処理部分 (ロジックをシンプルに修正) ---

        # ★修正点2: ファイル読み込みと初期化をまとめる
        try:
            # まずファイルの読み込みを試みる
            with open(file_name, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
        except FileNotFoundError:
            # ファイルがなければ、空の辞書からスタートする
            print(f"'{file_name}'が見つかりませんでした。新規作成します。")
            save_data = {}

        # ★修正点3: データを更新する処理をここに集約
        # 既存データ(save_data)に新しいデータ(update_data)を追加
        save_data.update(update_data)

        # ★修正点4: 書き込み処理を最後に一度だけ行う
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)
            print("jsonファイルへの出力が完了しました")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        if 'response' in locals():
            print("---APIからの生の応答---")
            print(response.text)

# --- 関数を実行 ---
if __name__ == "__main__":
    file_name = 'output_English_Sentence.json'
    gemini(10, "仮定法", file_name)

