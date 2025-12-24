import spacy
import json
import sys
from DB_hantei_kakunou import PatternRegistry

# ★モデルをロード (trfモデル推奨)
nlp = spacy.load("en_core_web_trf")

class Relative_Pronoun:

    def __init__(self, text, root_depth):
        self.text = text
        self.root_depth = root_depth 

    # ■ ヘルパー関数: 節のルート（動詞）を探し出す ■
    def get_clause_root(self, token):
        curr = token.head
        for _ in range(10):
            # 1. 明らかな節のルート
            if curr.dep_ in ["relcl", "ccomp", "advcl", "acl"] and curr.pos_ in ["VERB", "AUX"]: 
                return curr
            
            # 2. 文脈によってルートになりうるもの (動詞に限る)
            # 名詞(pobj)で止まらず、さらに上の動詞を探しに行く
            if curr.dep_ in ["pobj", "csubj", "nsubj", "dobj", "attr"] and curr.pos_ in ["VERB", "AUX"]:
                return curr

            if curr == curr.head: 
                break
            curr = curr.head
        
        return token.head
    
    def whats_syuusyoku(self,token):
        Syuusyoku = token.head.pos_

        if Syuusyoku in ["NOUN","PROPN"]:
            return "名詞修飾"
        
        elif Syuusyoku == "PRON":
            return "代名詞修飾"
        
        elif Syuusyoku =="VERB":
            return "文修飾"
        
        else:
            return f"修飾関係不明(.head->{Syuusyoku})"

    def serch_kankei(self):
        doc = nlp(self.text)
        tags = ["WP", "WDT", "WP$"]
        tags_hukushi = ["WRB"]
        end_token = []
      

        found_items = []
        result_token = []

        for words in doc[1:]:
            j = 0

            # --- Block 1: 'that' の処理 ---
            if words.text.lower() == "that":
                j = 1 
                head_verb = self.get_clause_root(words)
                Syuusyoku_TEXT = self.whats_syuusyoku(head_verb)
                
                # パターン1: 主格
                if words.dep_ in ["nsubj", "nsubjpass","attr"]:
                    if head_verb.dep_ in ["relcl", "acl"]:
                        end_token.append(f"関係代名詞(主格), 修飾種別 => {Syuusyoku_TEXT}")
                        found_items.append(head_verb.i)
                        result_token.append((head_verb.i, f"関係代名詞(主格), 修飾種別 => {Syuusyoku_TEXT}"))

                # パターン2: 目的格
                elif words.dep_ in ["dobj", "pobj", "dative"]:
                    if head_verb.dep_ in ["relcl", "acl"]: 
                        end_token.append(f"関係代名詞(目的格), 修飾種別 => {Syuusyoku_TEXT}")
                        found_items.append(head_verb.i)
                        result_token.append((head_verb.i, f"関係代名詞(目的格), 修飾種別 => {Syuusyoku_TEXT}"))

                # パターン3: 接続詞的用法 (mark) -> 同格の可能性が高い
                elif words.dep_ == "mark":
                    verb_head = words.head 
                    grand_head = verb_head.head 

                    # 親が名詞で、かつ従属節(acl/appos/ccomp)として機能しているなら同格
                    if grand_head.pos_ in ["NOUN", "PROPN"] and verb_head.dep_ in ["acl", "appos", "ccomp"]:
                        #end_token.append("同格thatです。関係代名詞ではありません")
                        print("同格thatです。関係代名詞ではありません")
                    
                    # mark判定されているが、構造上relclであれば関係代名詞とする
                    elif verb_head.dep_ in ["relcl"]:
                        Syuusyoku_TEXT2 = self.whats_syuusyoku(verb_head)
                        end_token.append(f"関係代名詞(目的格), 修飾種別 => {Syuusyoku_TEXT2}")
                        found_items.append(verb_head.i)
                        result_token.append((verb_head.i, f"関係代名詞(目的格), 修飾種別 => {Syuusyoku_TEXT2}"))

                    else:
                        #end_token.append(f"接続詞that/不明 ({verb_head.dep_})")
                        print("接続詞that")

                else:
                    j = 0 

            # --- Block 2: 'that' 以外の処理 ---
            elif j == 0 and words.tag_ in ["WP", "WDT", "WRB", "WP$"]:
                
                # 1. Piping判定
                is_piping = False
                piping_prep_text = ""
                
                if words.i > 0:
                    prev_token = doc[words.i - 1]
                    if prev_token.pos_ == "ADP" or prev_token.text.lower() == "to":
                        if words.head == prev_token:
                            is_piping = True
                            piping_prep_text = prev_token.text
                        elif words.tag_ == "WP$" and words.head.head == prev_token:
                            is_piping = True
                            piping_prep_text = prev_token.text

                # 2. 節のルート探索
                current = self.get_clause_root(words)
                Syuusyoku_TEXT3 = self.whats_syuusyoku(current)
                # ★【修正点: パース断絶時の前方検索】
                # get_clause_rootで有効な節が見つからない場合(解析失敗時)、
                # 後ろにある一番近い動詞(relcl/acl)を強引に探しに行く
                if current.dep_ not in ["relcl", "acl", "ccomp", "advcl"]:
                    for k in range(words.i + 1, len(doc)):
                        if doc[k].dep_ in ["relcl", "acl"] and doc[k].pos_ in ["VERB", "AUX"]:
                            current = doc[k]
                            break
                
                clause_dep = current.dep_

                # 3. カンマ判定
                iskanma = False
                check_index = words.i - 1 if not is_piping else words.i - 2
                if check_index >= 0 and doc[check_index].text == ",":
                    iskanma = True

                # 4. 最終分類
                
                # 間接疑問文
                if clause_dep in ["ccomp", "csubj", "pobj"] and current.pos_ in ["VERB", "AUX"]:
                    label = f"間接疑問文: {words.text}"
                    if is_piping:
                         label = f"間接疑問文(前置詞付き)"
                    #end_token.append(label)
                    j = 1

                # 関係詞節
                elif clause_dep in ["relcl", "advcl", "acl"]:
                    
                    if words.tag_ in tags_hukushi:
                        end_token.append(f"関係副詞 : {words.text}")
                        found_items.append(current.i)
                        result_token.append((current.i, f"関係副詞 : {words.text}"))
                        j = 1

                    elif words.tag_ in tags:
                        j = 1
                        label = ""
                        if is_piping:
                            if words.tag_ == "WP$":
                                label = f"Formal関係代名詞(所有格・前置詞付き), 修飾種別 => {Syuusyoku_TEXT3}"
                                
                            else:
                                label = f"Formal関係代名詞(目的格・前置詞付き), 修飾種別 => {Syuusyoku_TEXT3}"
                        else:
                            prefix = ""
                            if words.head.pos_ == "ADP" and words.head.i > words.i:
                                prefix = "Casual"
                            
                            if words.tag_ == "WP$": label = f"{prefix}関係代名詞(所有格), 修飾種別 => {Syuusyoku_TEXT3}"
                            elif words.dep_ in ["nsubj", "nsubjpass"]: label = f"{prefix}関係代名詞(主格), 修飾種別 => {Syuusyoku_TEXT3}"
                            elif words.dep_ in ["dobj", "pobj", "dative", "mark"]: label = f"{prefix}関係代名詞(目的格), 修飾種別 => {Syuusyoku_TEXT3}"
                            elif words.dep_ == "poss": label = f"{prefix}関係代名詞(所有格), 修飾種別 => {Syuusyoku_TEXT3}"
                            else: label = f"{prefix}関係代名詞(不明・副詞的用法: {words.dep_}), 修飾種別 => {Syuusyoku_TEXT3}"

                        if iskanma:
                             label += "!-非制限用法-!"

                        end_token.append(f"{label}")
                        found_items.append(current.i)
                        result_token.append((current.i, f"{label}"))

        # --- Block 3: 省略形 (Omission) の判定 ---
        
        for words in doc[1:]:        
            if words.dep_ in ["relcl", "acl"] and not words.pos_ == "NOUN": 
                if words.dep_ == "acl" and words.pos_ not in ["VERB", "AUX"]:
                    continue

                maker = False
                if words.i in found_items:
                    maker = True
                
                has_subject = False
                for child in words.children:
                    if child.dep_ in ["nsubj", "nsubjpass","expl"]:
                        has_subject = True
                        break
                
                if not maker and has_subject :
                    has_mark = False
                    for child in words.children:
                         if child.dep_ == "mark":
                             has_mark = True
                             break
                    
                    if not has_mark:
                        Syuusyoku_TEXT4 = self.whats_syuusyoku(words)
                        end_token.append(f"関係代名詞(省略形), 修飾種別 => {Syuusyoku_TEXT4}")
                        result_token.append((words.i, f"関係代名詞(省略形), 修飾種別 => {Syuusyoku_TEXT4}"))
        
        if end_token == [] and result_token == []:
            return ["なし"]
        
        elif result_token == []:
            return end_token
        
        else :
            result_token.sort(key = lambda x: x[0])
            end_token = [item[1] for item in result_token]
            return end_token






def main(testdata):
    print("-" * 30)
    print(f"原文: {testdata}")
    # クラスのインスタンス化
    registry = PatternRegistry()
    root_depth = 6
    
    # 解析実行
    test_text_serch = Relative_Pronoun(testdata, root_depth)
    found_list = test_text_serch.serch_kankei() # ここでリストが返ってくる
    
    # 結果表示（デバッグ用）
    print(f"解析結果リスト: {found_list}")

    pattern_ids = []

    # ★修正ポイント: リストの中身を一つずつ取り出してIDを取得する
    if found_list:
        for result_text in found_list:
            # 個別の判定結果（文字列）を渡してIDをもらう
            p_id = registry.get_id(result_text) # 引数名はPatternRegistryの実装に合わせてください
            
            # ★修正ポイント: appendの結果を代入しない
            pattern_ids.append(p_id)
            print(f"  -> '{result_text}' のIDは {p_id} です")
    else:
        print("  -> 関係詞は見つかりませんでした")

    # 最終的なIDのリストを表示・返却
    print(f"最終IDリスト: {pattern_ids}")
    
    
    return pattern_ids

if __name__ == "__main__":
    # テスト実行
    main("I was a Child which see the sea everyday.")
    main("The Teacher which I like is very busy.")
    # 3つの関係詞が含まれる文
    main("The woman who is talking to the man I respect is the teacher whose class I take.")
