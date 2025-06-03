import os
import csv

# 輸入(list:[要融合的資料集,,], str: csv_file_path)
def create_csv(dataset_dirs, csv_file_path):
    # 保存資料集資訊的列表
    dataset_info = []
    for dataset_dir in dataset_dirs:
        # 獲取所有類別子資料夾的路徑
        class_folders = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]
        
        # 遍歷每個類別子資料夾
        for class_folder in class_folders:
            class_path = os.path.join(dataset_dir, class_folder)
            
            # 獲取類別子資料夾中的圖片列表
            image_files = [file for file in os.listdir(class_path) if file.endswith('.jpg') or file.endswith('.png')]
            
            # 將每個圖片的路徑和對應的類別標籤添加到資料集資訊列表中
            for image_file in image_files:
                image_path = os.path.join( dataset_dir, class_folder, image_file)
                dataset_info.append((image_path, image_file, class_folder))

    
    # 將資料集資訊寫入 CSV 檔案
    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'image_name', 'label'])
        writer.writerows(dataset_info)


# 資料集目錄
dataset_directory = [
    'datasets/tieredImagenet/train',
]
# 建立的 CSV 檔案路徑
csv_file_path = 'datasets/tieredImagenet_train.csv'

# 呼叫函數創建 CSV 檔案
create_csv(dataset_directory, csv_file_path)