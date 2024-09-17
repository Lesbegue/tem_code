import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
def get_data(dt):
    if dt == 'accelerometry-walk-climb-drive':
        signal = pd.read_csv('data/accelerometry-walk-climb-drive/id00b70b13.csv', nrows=3000)[['lw_x', 'la_x']]
        time = pd.read_csv('data/accelerometry-walk-climb-drive/id00b70b13.csv', nrows=3000)['time_s']
        freq = 0.5
    elif dt == 'brain-wearable-monitoring':
        signal = pd.concat([
            pd.read_csv('data/brain-wearable-monitoring/Left_HR.csv', nrows=3000)['HR'],
            pd.read_csv('data/brain-wearable-monitoring/Right_HR.csv', nrows=3000)['HR']
        ], axis=1)
        signal.columns = ['Left_HR', 'Right_HR']
        time = pd.Series(range(len(signal)))
        freq = 0.03
        # if signal is not None:
        #     print(signal.head())
        #     print(time)
    elif dt == 'consumer-grade-wearables':
        signal = pd.concat(
            [pd.read_csv(f'data/consumer-grade-wearables/empatica_bvp.csv')['bvp'],
             pd.read_csv(f'data/consumer-grade-wearables/empatica_eda.csv')['eda']], axis=1)[
                 :min(len(pd.read_csv(f'data/consumer-grade-wearables/empatica_bvp.csv')['bvp']),
                      len(pd.read_csv(f'data/consumer-grade-wearables/empatica_eda.csv')['eda']))]
        time = pd.Series(range(len(signal)))
        signal.columns = ['bvp', 'eda']
        freq = 0.03
        # if signal is not None:
        #     print(signal.head())
        #     print(time)
    elif dt == 'Heart_Rate':
        signal = pd.concat([pd.read_csv('data/Heart_Rate/hr1.txt', header=None),
                            pd.read_csv('data/Heart_Rate/hr2.txt', header=None)], axis=1, ignore_index=True)

        time = torch.tensor([i for i in range(signal.shape[0])], dtype=torch.float32)
        signal.columns = ['hr1', 'hr2']
        freq = 0.03
        # if signal is not None:
        #     print(signal.head())
        #     print(time)
    elif dt == 'culm':
        signal = pd.concat(
            [pd.read_csv(f'data/culm/BostonCA01.csv')['X'],
             pd.read_csv(f'data/culm/BostonCA02.csv')['X']], axis=1)[
                 :min(len(pd.read_csv(f'data/culm/BostonCA01.csv')['X']),
                      len(pd.read_csv(f'data/culm/BostonCA02.csv')['X']))]
        signal.columns = ['BostonCA01_X', 'BostonCA02_X']
        time = pd.Series(range(len(signal)))
        freq = 0.03
        # if signal is not None:
        #     print(signal.head())
    elif dt == 'cpap-data-canterbury':
        signal = pd.read_csv('data/cpap-data-canterbury/Subject_01_4cmH20_short_breaths.csv',nrows=3000)[['Pressure at Venturi 1 [cmH2O]','Flow at Venturi 1 [L/s]']]
        signal.columns = ['Pressure', 'Flow']
        time = pd.Series(range(len(signal)))
        freq = 0.03
        # if signal is not None:
        #     print(signal.head())
        #     print(time)
    elif dt == 'emgdb':
        signal = pd.concat([
            pd.read_csv('data/emgdb/emg_healthy.txt', header=None, nrows=3000, encoding='latin1', delimiter=r'\s+',
                        engine='python')[1],
            pd.read_csv('data/emgdb/emg_myopathy.txt', header=None, nrows=3000, encoding='latin1', delimiter=r'\s+',
                        engine='python')[1]
        ], axis=1, ignore_index=True).iloc[:min(len(
            pd.read_csv('data/emgdb/emg_healthy.txt', header=None, encoding='latin1', nrows=3000, delimiter=r'\s+',
                        engine='python')[1]),
            len(pd.read_csv('data/emgdb/emg_myopathy.txt', header=None, nrows=3000,
                            encoding='latin1', delimiter=r'\s+', engine='python')[
                    1]))]
        signal = signal.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
        time = pd.Series(range(len(signal)))
        signal.columns = ['healthy', 'myopathy']
        freq = 0.03
        # if signal is not None:
        #     print(signal.head())
        #     print(time)
    elif dt == 'gait-maturation-db':
        signal = pd.concat([
            pd.read_csv('data/gait-maturation-db/1-40.txt', header=None, encoding='latin1', delimiter=r'\s+',
                        engine='python')[1],
            pd.read_csv('data/gait-maturation-db/3-47.txt', header=None, encoding='latin1', delimiter=r'\s+',
                        engine='python')[1]
        ], axis=1, ignore_index=True).iloc[:min(len(
            pd.read_csv('data/gait-maturation-db/1-40.txt', header=None, encoding='latin1', delimiter=r'\s+',
                        engine='python')[1]),
            len(pd.read_csv('data/gait-maturation-db/3-47.txt', header=None,
                            encoding='latin1', delimiter=r'\s+', engine='python')[
                    1]))]
        signal.columns = ['1-40', '3-47']
        time = pd.Series(range(len(signal)))
        freq = 0.03
        # if signal is not None:
        #     print(signal.head())
        #     print(time)
    elif dt == 'perg-ioba-datasetb':
        signal = pd.read_csv('data/perg-ioba-dataset/0001.csv')[['RE_1','LE_1']]
        time = pd.Series(range(len(signal)))
        freq = 0.05
        # if signal is not None:
        #     print(signal.head())
        #     print(time)
    elif dt == 'respiratory-heartrate-dataset':
        signal = pd.read_csv('data/respiratory-heartrate-dataset/ProcessedData_Subject01_FEM.csv',nrows = 3000)[['PPG0','PPG1']]
        time = pd.Series(range(len(signal)))
        freq = 0.03
        # if signal is not None:
        #     print(signal.head())
        #     print(time)
    elif dt == 'treadmill-exercise-cardioresp':
        signal = pd.read_csv('data/treadmill-exercise-cardioresp/test_measure.csv',nrows = 3000)[['VO2','VCO2']]
        time = pd.Series(range(len(signal)))
        freq = 0.03
        # if signal is not None:
        #     print(signal.head())
        #     print(time)
    elif dt == 'sinus-rhythm-dataset':
        time = pd.Series(np.arange(0, 1800, 1))
        # 生成多列信号数据（例如两个信号通道）
        signal1 = np.sin(0.01 *2*torch.pi* time)
        signal2 = np.cos(0.03 *2*torch.pi* time)  # 另一个信号通道

        # 创建 DataFrame
        df = pd.DataFrame({
            'time': time,
            'signal1': signal1,
            'signal2': signal2
        })
        time = df['time']
        signal = df[['signal1','signal2']]
        freq = 0.04
    if not isinstance(time, torch.Tensor):
        time = torch.tensor(time.values, dtype=torch.float32)
    return freq, time, signal,signal.columns

def sample_and_normalize_data(time, signal, size=1/3):
    # 确保 time 和 signal 都是张量
    assert isinstance(time, torch.Tensor), "time must be a torch.Tensor"
    assert isinstance(signal, torch.Tensor), "signal must be a torch.Tensor"

    # 计算抽样大小
    num_samples = int(len(time) * size)

    # 随机抽样索引
    indices = np.sort(np.random.choice(len(time), size=num_samples, replace=False))

    # 根据索引进行抽样
    sampled_time = time[indices]
    sampled_signal = signal[indices]

    # 计算 sampled_time 的均值
    time_mean = sampled_time.mean()

    # 去均值化
    sampled_time_centered = sampled_time - time_mean

    # 计算 sampled_signal 的均值
    signal_mean = sampled_signal.mean(dim=0)  # 对于多维信号，计算每列的均值
    sampled_signal_centered = sampled_signal - signal_mean

    return sampled_time_centered, sampled_signal_centered

def main():
    dt_options = [
        'accelerometry-walk-climb-drive',
        'brain-wearable-monitoring',
        'consumer-grade-wearables',
        'Heart_Rate',
        'culm',
        'cpap-data-canterbury',
        'emgdb',
        'gait-maturation-db',
        'perg-ioba-datasetb',
        'respiratory-heartrate-dataset',
        'treadmill-exercise-cardioresp',
        'sinus-rhythm-dataset'
    ]
    for dt in dt_options:
        time,signal,columns = get_data(dt)
        print(columns)
        print(len(time) == len(signal))



if __name__ == '__main__':
    main()
