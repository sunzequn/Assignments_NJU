import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import json
import csv
from rec_date_clean import *


renta_dict = {'ALBACETE': 76895, 'ALICANTE': 60562, 'ALMERIA': 77815, 'ASTURIAS': 83995, 'AVILA': 78525,
              'BADAJOZ': 60155, 'BALEARS, ILLES': 114223, 'BARCELONA': 135149, 'BURGOS': 87410, 'NAVARRA': 101850,
              'CACERES': 78691, 'CADIZ': 75397, 'CANTABRIA': 87142, 'CASTELLON': 70359, 'CEUTA': 333283,
              'CIUDAD REAL': 61962, 'CORDOBA': 63260, 'CORUÑA, A': 103567, 'CUENCA': 70751, 'GIRONA': 100208,
              'GRANADA': 80489,
              'GUADALAJARA': 100635, 'HUELVA': 75534, 'HUESCA': 80324, 'JAEN': 67016, 'LEON': 76339,
              'LERIDA': 59191, 'LUGO': 68219, 'MADRID': 141381, 'MALAGA': 89534, 'MELILLA': 116469,
              'GIPUZKOA': 101850,
              'MURCIA': 68713, 'OURENSE': 78776, 'PALENCIA': 90843, 'PALMAS, LAS': 78168, 'PONTEVEDRA': 94328,
              'RIOJA, LA': 91545, 'SALAMANCA': 88738, 'SANTA CRUZ DE TENERIFE': 83383, 'ALAVA': 101850,
              'BIZKAIA': 101850,
              'SEGOVIA': 81287, 'SEVILLA': 94814, 'SORIA': 71615, 'TARRAGONA': 81330, 'TERUEL': 64053,
              'TOLEDO': 65242, 'UNKNOWN': 103689, 'VALENCIA': 73463, 'VALLADOLID': 92032, 'ZAMORA': 73727,
              'ZARAGOZA': 98827}

# 'fecha_dato'与'ncodpers'单独处理
all_features = ['fecha_dato', 'ncodpers', 'ind_empleado', 'pais_residencia', 'sexo',
                'age', 'fecha_alta', 'ind_nuevo', 'antiguedad', 'indrel',
                'ult_fec_cli_1t', 'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext',
                'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'cod_prov',
                'nomprov', 'ind_actividad_cliente', 'renta', 'segmento']


bad_features = list(set(all_features).difference(set(good_features)))

# 去掉了'ind_ahor_fin_ult1', 'ind_aval_fin_ult1',
products_list = [
    # 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1',
    'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
    'ind_ctju_fin_ult1',
    'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
    'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
    'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
    'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
    'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

continuous_features = ['age', 'antiguedad', 'renta']
discrete_features = list(set(good_features) ^ set(continuous_features))
print("连续特征: ", continuous_features)
print("离散特征: ", discrete_features)

nan_list = ['NA', '']


def handle_age(row, age_min=18, age_max=100, age_nan=40):
    age = age_nan
    if row["age"].strip() not in nan_list:
        age = float(row["age"].strip())
        age = max(age, age_min)
        age = min(age, age_max)
    return round((age - age_min) / (age_max - age_min), 6)


def handle_antiguedad(row, antiguedad_min=0, antiguedad_max=256, antiguedad_nan=0):
    antiguedad = antiguedad_nan
    if row['antiguedad'].strip() not in nan_list:
        antiguedad = float(row['antiguedad'].strip())
        antiguedad = max(antiguedad, antiguedad_min)
        antiguedad = min(antiguedad, antiguedad_max)
    return round((antiguedad - antiguedad_min) / (antiguedad_max - antiguedad_min), 6)


def handle_renta(row, renta_min=0, renta_max=1500000):
    rent = row['renta'].strip()
    if rent in nan_list:
        if row['nomprov'].strip() in nan_list:
            rent = float(renta_dict['UNKNOWN'])
        else:
            rent = float(renta_dict[row['nomprov']])
    else:
        rent = float(rent)
        rent = max(rent, renta_min)
        rent = min(rent, renta_max)
    return round((rent - renta_min) / (renta_max - renta_min), 6)


def handle_month(row):
    return int(row['fecha_dato'].strip().split('-')[1])


def load_mapping():
    f = open('mapping', 'r')
    lines = ""
    for line in f:
        lines += line
    mapping = json.loads(lines)
    print("加载mapping文件", type(mapping))
    return mapping


def gene_features(row, one_hot_mapping_dict):
    features = [handle_age(row), handle_antiguedad(row), handle_renta(row), handle_month(row)]
    for f in discrete_features:
        v = row[f].strip()
        if v in one_hot_mapping_dict[f].keys():
            feature = one_hot_mapping_dict[f].get(v)
        else:
            feature = one_hot_mapping_dict[f].get("NAN", [0] * len(one_hot_mapping_dict[f]))
            # print(f, "异常:" + v)
        if len(feature) < 2:
            print("error")
        features = features + feature
    return features


def gene_products(row):
    prods = []
    for prod in products_list:
        p = 0
        if row[prod].strip() not in nan_list:
            p = int(row[prod].strip())
        prods.append(p)
    return prods


def process_train_data(train_file, one_hot_mapping):
    print("开始构造训练数据...")
    t = datetime.datetime.now()
    train_list = []
    train_label = []
    num = 0
    f = True
    user_products_dict = {}
    dates = set()

    for row in csv.DictReader(train_file):
        num += 1
        if num % 2000000 == 0:
            print("已处理: ", num)

        usr = int(row['ncodpers'])
        date = row['fecha_dato'].strip()
        dates.add(date)
        if len(dates) > 1:
            prev_products = user_products_dict.get(usr, [0] * len(products_list))
            post_products = gene_products(row)
            new_products = [max(x1 - x2, 0) for (x1, x2) in zip(post_products, prev_products)]
            if sum(new_products) > 0:
                for ind, prod in enumerate(new_products):
                    if prod > 0:
                        features = gene_features(row, one_hot_mapping)
                        if f:
                            print("训练数据特征数量: ", len(features), len(prev_products))
                            f = False
                        train_list.append(features + prev_products)
                        train_label.append(ind)

        user_products_dict[usr] = gene_products(row)

    print("日期", dates)
    print("构造训练数据耗时: " + str(datetime.datetime.now() - t))
    return train_list, train_label, user_products_dict


def process_test_data(test_file, user_products_dict, one_hot_mapping):
    print("开始构造测试数据...")
    t = datetime.datetime.now()
    test_list = []
    num = 0
    f = True
    for row in csv.DictReader(test_file):
        num += 1
        if num % 500000 == 0:
            print("已处理: ", num)

        usr = int(row['ncodpers'])
        prev_products = user_products_dict.get(usr, [0] * len(products_list))
        features = gene_features(row, one_hot_mapping)
        if f:
            print("测试数据特征数量: ", len(features), len(prev_products))
            f = False
        test_list.append(features + prev_products)
    print("构造测试数据耗时: " + str(datetime.datetime.now() - t))
    return test_list


def xgb_model(train_X, train_y, seed_val=0):
    param = {'objective': 'multi:softprob', 'eta': 0.05, 'max_depth': 6, 'silent': 1, 'num_class': 22,
             'eval_metric': "mlogloss", 'min_child_weight': 2, 'subsample': 0.9, 'colsample_bytree': 0.9,
             'seed': seed_val}
    num_rounds = 200
    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    watchlist = [(xgtrain, 'train')]
    model = xgb.train(plst, xgtrain, num_rounds, evals=watchlist)
    return model


def rec(train_file, test_file, res_file):
    one_hot_mapping_dict = load_mapping()
    train_list, train_label, user_products_dict = process_train_data(train_file, one_hot_mapping_dict)
    test_list = process_test_data(test_file, user_products_dict, one_hot_mapping_dict)

    train_data = np.array(train_list)
    train_label = np.array(train_label)
    print("训练数据维数: ", train_data.shape, train_label.shape)

    test_data = np.array(test_list)
    print("测试数据维数: ", test_data.shape)

    print("训练模型..")
    model = xgb_model(train_data, train_label, seed_val=0)
    del train_data, train_label
    print("预测产品..")
    xgtest = xgb.DMatrix(test_data)
    preds = model.predict(xgtest)
    del test_data, xgtest

    print("选择结果..")
    target_cols = np.array(products_list)
    preds = np.argsort(preds, axis=1)
    preds = np.fliplr(preds)[:, :8]
    test_id = np.array(pd.read_csv(test_file_path, usecols=['ncodpers'])['ncodpers'])
    final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
    out_df = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
    out_df.to_csv(res_file, index=False)


if __name__ == '__main__':
    train_file = open("../train_ver2.csv")
    test_file = open('../test_ver2.csv')
    test_file_path = "../test_ver2.csv"
    res_file = 'rec_all_new.csv'
    t = datetime.datetime.now()
    rec(train_file, test_file, res_file)
    print("总耗时: ", datetime.datetime.now() - t)
