import pandas as pd
import numpy as np

def get_fri_df(basepath):
    return pd.read_csv(basepath + 'fri_data_v2.csv')

def get_risk_factors():
    maternal_risks = ["htn", "anemia", "asthma", "obesity", "smoker", "ama", "teen", "lpnc", "drugs", "cholesterol", "gdm", "dm", "short"]
    obstetrical_risks = ["arom", "prom", "srom", "iugr", "postdate", "protract", "arrest"]
    fetal_risks = ["mecon", "chorio", "brady"]
    delivery_risks = ["abnfhrtotal","abnvartotal","absentacceltotal", "abndeceltotal","exuatotal", "oligo", "cs", "parity", "weeks", "weightgrams", "laborhours"]
    return maternal_risks, obstetrical_risks, fetal_risks, delivery_risks

def print_risk_factors():
    m, o, f, d = get_risk_factors()
    print(f'Maternal risks ({len(m)}):', m)
    print(f'Obstetrical risks ({len(o)}):', o)
    print(f'Fetal risks ({len(f)}):', f)
    print(f'Delivery risks ({len(d)}):', d)
    print(f'Total risk factors: ({len(m) + len(o) + len(f) + len(d)})')

def get_categorical_columns():
    m, o, f, d = get_risk_factors()
    cat_feat = list(range(23)) + [28, 29]
    num_feat = list(range(23, 28)) + list(range(30, 34))
    return np.array((m+o+f+d))[cat_feat]