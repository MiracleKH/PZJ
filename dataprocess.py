import os
import pandas as pd

root_dir = r'G:\pythonfiles\pzj\pjzpython\data'

if not os.path.exists(root_dir):
    print(f"错误警告：根目录 {root_dir} 不存在，请检查路径是否正确。")
else:

    for main_folder in ['Tiptoe', 'Squat', 'Run', 'Stand', 'Walk']:
        main_path = os.path.join(root_dir, main_folder)


        if not os.path.exists(main_path):
            print(f"错误警告：主目录 {main_path} 不存在，跳过该主目录。")
            continue

        # 初始化一个空的DataFrame来存储主文件夹的所有数据
        main_merged_data = pd.DataFrame()

        for sub_folder in ['Back', 'Right', 'Left']:
            folder_name =sub_folder
            folder_path = os.path.join(main_path, folder_name)


            if not os.path.exists(folder_path):
                print(f"错误警告：子目录 {folder_path} 不存在，跳过该子目录。")
                continue

            # 初始化一个空的Series来存储当前子文件夹的数据
            subfolder_data = pd.Series(dtype=float)

            # 遍历sensor文件
            for i in range(1, 10):
                file_name = f'sensor ({i}).csv'
                file_path = os.path.join(folder_path, file_name)
                if not os.path.exists(file_path):
                    print(f"错误警告：文件 {file_path} 不存在，跳过该文件。")
                    continue

                # 读取CSV文件
                try:
                    data = pd.read_csv(file_path, skiprows=8, header=None, usecols=[0])

                    if subfolder_data.empty:
                        subfolder_data = data.squeeze()  # 转换为Series
                    else:
                        subfolder_data = pd.concat([subfolder_data, data.squeeze()], ignore_index=True)
                except Exception as e:
                    print(f"错误警告：处理文件 {file_path} 时出错，错误信息为：{e}")


            if subfolder_data.empty:
                print(f"错误警告：子文件夹 {folder_path} 没有有效数据，跳过该子文件夹。")
                continue

            # 将子文件夹数据添加到主DataFrame中
            main_merged_data[folder_name] = subfolder_data


        if main_merged_data.empty:
            print(f"错误警告：主文件夹 {main_path} 没有有效数据，跳过保存操作。")
            continue


        main_merged_data['Label'] = main_folder

        # 保存合并后的文件
        output_file = os.path.join(root_dir, f'{main_folder}.csv')
        try:
            main_merged_data.to_csv(output_file, index=False)
            print(f"已保存合并文件: {output_file}")
        except Exception as e:
            print(f"错误警告：保存文件 {output_file} 时出错，错误信息为：{e}")

print("所有文件合并流程结束！")