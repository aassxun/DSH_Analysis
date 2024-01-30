import re
import numpy as np
import csv
import os
def get_result(file_path):
    result = []
    with open(file_path,'r') as f:

        for line in f.readlines():
            line = line.strip()
            if '__main__:run:126' in line:
                number = re.findall(r"0\.\d{4}",line)
                number = [float(i) for i in number]
                result.append(number)
    result = np.array(result).T*100
    # print(result)
    reshaped_r = result.T.reshape(-1,2)
    joined = ['/'.join(map(lambda x: format(x, '.1f'), pair.flatten())) for pair in reshaped_r]
    joined = np.array(joined)    
    
    odd_rows =joined[::2]
    even_rows = joined[1::2]    
    rearranged_r = np.vstack((odd_rows, even_rows))
    # print(rearranged_r)    
    return  rearranged_r




filepaths = []
for dirpath, dirnames, filenames in os.walk('.'):
    for filename in filenames:
        filepaths.append(os.path.join(dirpath, filename))
# 创建或打开一个CSV文件
with open('results.csv', 'w', newline='') as csvfile:
    fieldnames = ['filepath', 'bits','result']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 写入表头
    writer.writeheader()

    for filepath in filepaths:
        #--------------------------------------------------------------------------------------
        if "log" in filepath and 'nus-wide' in filepath and '16-5' in filepath:
        #--------------------------------------------------------------------------------------
            print(filepath.split('\\')[-2])
            r = get_result(filepath)
            r = r.flatten()
            print(r)
            # 写入一行数据
            writer.writerow({'filepath': filepath.split('\\')[-2].split('-')[4], #3 4  7
                             'bits': filepath.split('\\')[-1].replace('.log', ''),
                             'result': str(r).replace('\n', '').replace('[', '').replace(']', '')})