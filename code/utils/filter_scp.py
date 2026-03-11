import pandas as pd
import ast
import os

def filtering_scp(path):
    #Filtering for train, val and test:
    df = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    dg = pd.read_csv(os.path.join(path, 'scp_statements.csv'), index_col = 0)
    agg_df = dg[dg.diagnostic == 1]
            
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        if tmp:
            return tmp[0]
        else:
            return 'nicht_bestatigt'

    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)

    df_new = df[df.diagnostic_superclass != 'nicht_bestatigt']
    df_new = df_new[df_new.validated_by_human == 1]
    
    return df_new