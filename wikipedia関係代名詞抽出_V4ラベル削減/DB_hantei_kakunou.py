import json
import os

class PatternRegistry:
    def __init__(self, db_path='pattern_db.json'):
        self.db_path = db_path
        self.data = self._load_db()

    def _load_db(self):
        if not os.path.exists(self.db_path):
            return {}
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def get_id(self, pattern_text):
        # メモリ上のデータを確認
        if pattern_text in self.data:
            return self.data[pattern_text]
        
        # 新規ID発行
        new_id = max(self.data.values()) + 1 if self.data else 0
            
        # メモリ更新 & ファイル保存
        self.data[pattern_text] = new_id
        self._save_db()
        
        return new_id

    def _save_db(self):
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)