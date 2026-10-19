import pandas as pd
import numpy as np
import json
import os
import pickle
from sklearn.model_selection import train_test_split

def create_baseline_data():
    # 1. 設定檔初始化 (模擬原專案的 config)
    config = {
        'features_to_extract': [
            'roundscore_A', 'roundscore_B', 'player_id', 'opponent_id',
            'type', 'backhand', 'location', 'strength', 'spin', 'set'
        ],
        'max_shot_seq_len': 35,
        'max_rally_seq_len': 30,
        'max_set_seq_len': 7,
    }
    output_dir = "processed_baseline_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 讀取雙打資料 (⚠️ 請確保這裡的檔名和您的 CSV 一模一樣)
    input_file = "dataset_doubles_all.csv" 
    
    if not os.path.exists(input_file):
        print(f"找不到檔案 '{input_file}'，請確認檔名是否正確，且與此腳本放在同一個資料夾。")
        return
        
    matches = pd.read_csv(input_file)
    
    # 3. 🎯 修正後的欄位對應 (Data Mapping)
    rename_mapping = {
        'scoreSelf': 'roundscore_A',        # 主視角得分
        'scoreOther': 'roundscore_B',       # 對側得分
        'numberGame': 'set',                # 局數
        'actionId': 'type',                 # ⚠️ 修正：具體的擊球動作類型
        'handId': 'backhand',               # 正反手
        'pointId': 'location',              # 落點
        'strengthId': 'strength',           # 力道
        'spinId': 'spin',                   # 旋轉
        'gamePairId': 'player_id',        # 當下擊球者 ID
        'gamePairOtherId': 'opponent_id', # 上一拍擊球者 (強行當作單打對手)
        'match': 'match_id',                # 比賽 ID
        'strikeNumber': 'ball_round'        # 擊球順序
    }
    matches = matches.rename(columns=rename_mapping)
    
    # 4. 類別特徵加 1 (原作者邏輯：為 Embedding 的 padding 預留 0)
    categorical_features_plus_one = ['type', 'backhand', 'location', 'strength', 'spin']
    for col in categorical_features_plus_one:
        # 確保資料為整數型態再加 1
        matches[col] = pd.to_numeric(matches[col], errors='coerce').fillna(0).astype(int) + 1
        config[f'num_{col}'] = int(matches[col].max() + 1)
        
    # 5. 重新編碼 Player ID (讓它變成從 1 開始的連續整數)
    all_players = pd.concat([matches['player_id'], matches['opponent_id']]).unique()
    player_map = {int(name): i + 1 for i, name in enumerate(all_players) if pd.notnull(name)}
    matches['player_id'] = matches['player_id'].map(player_map)
    matches['opponent_id'] = matches['opponent_id'].map(player_map)
    config['num_players'] = len(all_players) + 1
    
    # 6. 建立 MT-HTA 要求的階層式資料結構 (Match -> Set -> Rally -> Shots)
    features_to_extract = config['features_to_extract']
    all_matches_data = []
    
    for match_id, match_df in matches.groupby('match_id'):
        match_data = {'match_id': match_id, 'sets': []}
        for set_num, set_df in match_df.groupby('set'):
            set_data = {'set': set_num, 'rallies': []}
            for rally_id, rally_df in set_df.groupby('rally_id'):
                rally_df = rally_df.sort_values('ball_round').copy()
                rally_shots = rally_df[features_to_extract].to_numpy() # 轉成 numpy array
                set_data['rallies'].append({'rally_id': rally_id, 'shots': rally_shots})
            set_data['rallies'] = sorted(set_data['rallies'], key=lambda x: x['rally_id'])
            match_data['sets'].append(set_data)
        all_matches_data.append(match_data)
        
    # 7. 以 match 為單位分割 train/validation/test
    match_ids = matches['match_id'].unique()
    train_match_ids, temp_match_ids = train_test_split(match_ids, test_size=0.3, random_state=42)
    val_match_ids, test_match_ids = train_test_split(temp_match_ids, test_size=0.66, random_state=42)
    
    config['test_match_ids'] = [int(i) for i in test_match_ids]
    config['val_match_ids'] = [int(i) for i in val_match_ids]
    
    train_data = [m for m in all_matches_data if m['match_id'] in train_match_ids]
    test_data = [m for m in all_matches_data if m['match_id'] in test_match_ids]
    val_data = [m for m in all_matches_data if m['match_id'] in val_match_ids]
    
    # 8. 儲存成模型要的 .pkl 檔案
    with open(os.path.join(output_dir, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(output_dir, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    with open(os.path.join(output_dir, 'val_data.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
        
    with open(os.path.join(output_dir, 'player_map.json'), 'w', encoding='utf-8') as f:
        json.dump(player_map, f, indent=4, ensure_ascii=False)
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Baseline 資料處理完成！已儲存於 '{output_dir}'")

if __name__ == '__main__':
    create_baseline_data()