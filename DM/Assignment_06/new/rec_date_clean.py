# -*- coding: utf-8 -*-

import datetime
import numpy as np
import pandas as pd
import json

# 'fecha_dato'与'ncodpers'单独处理
# 'ult_fec_cli_1t', 'fecha_alta'单独处理
all_features = ['ind_empleado', 'pais_residencia', 'sexo',
                'age', 'ind_nuevo', 'antiguedad', 'indrel',
                'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext',
                'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'cod_prov',
                'nomprov', 'ind_actividad_cliente', 'renta', 'segmento']


continuous_features = ['age', 'antiguedad', 'renta']
discrete_features = list(set(all_features) ^ set(continuous_features))


def handle_nan(df_col, v, is_strip=False):
    if is_strip:
        return df_col.map(
            lambda x: v if pd.isnull(x) or str(x).strip() == 'NA' or str(x).strip() == '' else str(x).strip())
    else:
        return df_col.map(lambda x: v if pd.isnull(x) or str(x).strip() == 'NA' or str(x).strip() == '' else x)


def handle_age(df):
    print("开始处理 age...")
    # 格式问题，处理空格与字符串年龄
    df["age"] = handle_nan(df["age"], np.NaN)
    df["age"] = df["age"].map(lambda x: int(x.strip()) if (isinstance(x, str)) else x)
    age_mean = df["age"].mean()
    age_min = df["age"].min()  # 18
    age_max = df["age"].max()  # 100
    print("age最小值为: ", age_min)
    print("age最大值为: ", age_max)
    print("age均值为: ", age_mean)


def handle_antiguedad(df):
    print("开始处理 antiguedad...")
    df['antiguedad'] = handle_nan(df['antiguedad'], 0)
    df["antiguedad"] = df["antiguedad"].map(lambda x: int(x.strip()) if (isinstance(x, str)) else x)
    df['antiguedad'] = df['antiguedad'].astype(int)
    min_v = df['antiguedad'].min()
    max_v = df['antiguedad'].max()
    print("antiguedad 最小值为: ", min_v)
    print("antiguedad 最大值为: ", max_v)


def handle_renta(df):
    print("开始处理 renta...")
    # mean_value = df['renta'].mean(skipna=True)
    df['renta'] = handle_nan(df['renta'], 101850)
    df["renta"] = df["renta"].map(lambda x: x.strip() if (isinstance(x, str)) else x)
    df['renta'] = df['renta'].astype(float)
    min_value = df['renta'].min()
    max_value = df['renta'].max()
    mean_value = df['renta'].mean()
    print("renta 最小值为: ", min_value)
    print("renta 最大值为: ", max_value)
    print("renta 均值为: ", mean_value)


def handle_renta_mapping(df):
    provs = df['nomprov'].unique()
    renta_prov_mapping = {}
    for prov in provs:
        m = df.loc[(df.nomprov == prov), "renta"].mean(skipna=True)
        renta_prov_mapping[prov] = m
    print(renta_prov_mapping)
    mapping_json = json.dumps(renta_prov_mapping)
    f = open('renta_mapping', 'w')
    f.write(mapping_json)
    f.close()


def handle_discrete_feature(df):
    print("开始处理 离散值...")
    f_mapping = {}
    for f in discrete_features:
        # 特殊情况
        if f == "indrel_1mes":
            f_mapping[f] = {'NAN': 0, '1.0': 1, '1': 1, '2.0': 2, '2': 2, '3.0': 3, '3': 3, '4.0': 4, '4': 4, 'P': 5}
            continue
        # 处理缺失值: NA '' NaN
        df[f] = handle_nan(df[f], "NAN", is_strip=True)
        mapping = {}
        values = list(df[f].unique())
        values_str = set()
        for v in values:
            # 处理浮点型数据多一个.0
            if v.endswith(".0"):
                v = v.rstrip(".0")
                if v == "":
                    v = "0"
            values_str.add(v)
        values_str = list(values_str)
        for v in values_str:
            # one-hot
            # l = [0] * len(values)
            # l[values_str.index(v)] = 1
            # mapping[v] = l
            mapping[v] = values_str.index(v)
        f_mapping[f] = mapping
    print("离散值mapping字典: ")
    print(f_mapping)
    print("数量统计")
    for key in f_mapping.keys():
        print(key, len(f_mapping[key]))
    mapping_json = json.dumps(f_mapping)
    f = open('mapping', 'w')
    f.write(mapping_json)
    f.close()


def clean_train_data(file):
    # 读取数据
    t = datetime.datetime.now()
    df_orign = pd.read_csv(file)
    print("读取文件耗时: " + str(datetime.datetime.now() - t))

    print("开始进行数据预处理...")
    t = datetime.datetime.now()
    # 处理连续值，规格化
    handle_age(df_orign)
    handle_antiguedad(df_orign)
    handle_renta(df_orign)
    # 处理离散值
    handle_discrete_feature(df_orign)
    # 处理renta 映射
    handle_renta_mapping(df_orign)
    print("数据预处理耗时: " + str(datetime.datetime.now() - t))


if __name__ == '__main__':
    train_file = "train_ver2.csv"
    clean_train_data(train_file)
