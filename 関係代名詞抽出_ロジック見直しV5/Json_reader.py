import json

def read_json_sentence(sentence_num,sentence_part,file_name):
    
    
    with open(file_name, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)

    
    keys = loaded_data.keys()
    test = []

    if sentence_part in keys:
        print(len(loaded_data[sentence_part]))

        for i in range(sentence_num):
            if len(loaded_data[sentence_part]) == 0 or len(loaded_data[sentence_part]) <= i:
                break
            test.append(loaded_data[sentence_part][i])

        return test
    
    else :
        return f"現在登録されている文法は『{keys}』です。この中から選択してください。"
    





if __name__ == "__main__":
    file_name = 'output_English_Sentence.json'
    test = read_json_sentence(101,"現在完了形",file_name)
    print(test)





