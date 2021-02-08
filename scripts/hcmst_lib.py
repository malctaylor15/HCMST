import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pdb, ipdb


def get_cnt_pct(col):
    """
    Get counts and percentage counts for each unique value in column

    :param col: pd.Series - column of data
    :return:
    """
    cnts = col.value_counts()
    pcts = np.round(col.value_counts(normalize=True) * 100, 2)
    cnts_pcts = pd.concat([cnts, pcts], keys=['Counts', "Percentage"], axis=1)
    return (cnts_pcts)

##############################
### Preprocessing Function ###
##############################

def encode_OneHot(df):
    """
    Wrapper for one hot enconding for series or all columns in data frame
    :param df: pandas.DataFrame
    :return: one_hot_df
    """
    if type(df) == pd.DataFrame:
        names = df.columns
        index = df.index
    elif type(df) == pd.Series:
        names = [df.name]
        index = df.index
        df = df.values.reshape(-1, 1)
    else:
        print('Data is not dataframe or series... need method to name columns')
        return()

    encoder = OneHotEncoder().fit(df)
    encoder_cols = encoder.get_feature_names(names)
    one_hot_array = encoder.transform(df).toarray()
    one_hot_df = pd.DataFrame(one_hot_array, columns=encoder_cols
                                    , index=index)
    return(one_hot_df)

def create_rulefit_df2(baseline_attrs, new_target, test_size=0.33):
    """
    input raw data (independent and dependent attributes) and return train test dataset
    """
    # One hot encode and add in null field for independent attributes selected earlier
    base1 = baseline_attrs.apply(lambda x: x.astype('str')).fillna('NA')
    baseline_one_hot=encode_OneHot(base1)

    modeling_df = baseline_one_hot.join(new_target, how='inner')
    question_number = new_target.name
    X_train, X_test, y_train, y_test = train_test_split(
        modeling_df.drop([question_number], axis=1), modeling_df[question_number]
        , test_size=test_size, random_state=42)

    return (X_train, X_test, y_train, y_test)

def preprocess_target(target, percentage_cutoff=10):
    """
    Preprocess target by removing nulls and combining low percentage targets into a single group

    """
    # Drop missing
    target2 = target.dropna()
    # Combine low percentage fields
    sorted_vals = get_cnt_pct(target)
    print("Percentages: \n", sorted_vals)
    small_pcts = sorted_vals[sorted_vals['Percentage'] < percentage_cutoff]
    if small_pcts.shape[0] < 2:
        # Only 1 field with less than x percent - don't replace
        pass
    else:
        target2 = target2.replace(small_pcts.index.tolist(), str(small_pcts.index.tolist()))
        sorted_vals = get_cnt_pct(target2)
        print('\nNew target percentages: ', sorted_vals)

    repl_dict = {sorted_vals.index[x]: x for x in range(sorted_vals.shape[0])}
    target2 = target2.replace(repl_dict)

    return (target2, repl_dict)

########################################
### Quantitative Evaluation Functions
########################################

def get_decile_metrics(gb, target_name, score_name):
    output_dict = {}
    output_dict['count'] = gb.shape[0]
    output_dict['min_score'] = np.round(gb[score_name].min() * 100, 2)
    output_dict['max_score'] = np.round(gb[score_name].max() * 100, 2)
    output_dict['avg_score'] = np.round(gb[score_name].mean() * 100, 2)
    output_dict['# of Positive Responders'] = gb[target_name].sum()
    output_dict['% of Positive Responders'] = np.round(gb[target_name].mean() * 100, 2)

    series = pd.Series(output_dict)
    return (series)

def get_decile_attr_metrics(gb, attr):
    output_dict = {}
    if gb[attr].dtype == 'category':
        gb[attr] = gb[attr].astype('float')
    output_dict[attr + ' Sum'] = gb[attr].sum()
    output_dict[attr + ' Average'] = np.round(gb[attr].mean()*100, 2)
    series = pd.Series(output_dict)
    return (series)

def rollup_target_gb_metrics(df):
    total_resp = df['# of Positive Responders'].sum()
    avg_resp = np.round((total_resp / df['count'].sum()) * 100, 2)
    df['Cumulative Responders'] = df['# of Positive Responders'][::-1].cumsum()
    df['Cumulative Capture Rate'] = np.round((df['Cumulative Responders'] / total_resp) * 100, 2)
    name_act_avg_resp = 'Actual % Pos - Avg(' + str(avg_resp) + '%)'
    df[name_act_avg_resp] = df['% of Positive Responders'] - avg_resp
    df['Actual % Pos - Decile Avg Score'] = df['% of Positive Responders'] - df['avg_score']
    
    # Summary Metrics
    sum_metrics = {}
    sum_metrics['count'] = df['count'].sum()
    sum_metrics['min_score'] = df['min_score'].min()
    sum_metrics['max_score'] = df['max_score'].max()
    sum_metrics['avg_score'] = df['avg_score'].mean()
    sum_metrics['# of Positive Responders'] = df['# of Positive Responders'].sum()
    sum_metrics['% of Positive Responders'] = avg_resp
    sum_metrics['Cumulative Responders'] = df['Cumulative Responders'].max()
    sum_metrics['Cumulative Capture Rate'] = df['Cumulative Capture Rate'].max()
    sum_metrics[name_act_avg_resp] = df[name_act_avg_resp].mean()
    sum_metrics['Actual % Pos - Decile Avg Score'] = np.abs(df['Actual % Pos - Decile Avg Score']).max()

#     df.index = pd.Index(df.index, str)
    df = df.append(pd.Series(sum_metrics, name="Total"))
    return (df)

def decile_tbl(train_preds, temp_y_train, attrs=''):
    target_name = temp_y_train.name
    score_name = train_preds.name
    if type(attrs) == str:
        merged = pd.concat([train_preds, temp_y_train], axis=1)
    else:
        merged = pd.concat([train_preds, temp_y_train, attrs], axis=1)

    merged['decile_cuts'] = pd.qcut(merged[score_name].rank(method='first'), [x for x in np.arange(0, 1.1, 0.1)]
                                    , labels=[x for x in range(1, 11)])
    merged['decile_cuts'] = merged['decile_cuts'].astype(int)
    gb1 = merged.groupby('decile_cuts')

    tbl1 = gb1.apply(lambda x: get_decile_metrics(x, target_name, score_name))
    target_tbl = rollup_target_gb_metrics(tbl1)
    if type(attrs) == str:
        return (target_tbl)
    # Rank ordering of attributes
    attr_tbls = []
    for col in attrs.columns:
        temp = gb1.apply(lambda x: get_decile_attr_metrics(x, col))

        attr_sum = temp[col + ' Sum'].sum()
        attr_mean = np.round((attr_sum / merged.shape[0])*100, 2)
        temp = temp.append(pd.Series([attr_sum, attr_mean]
                                     , index=[col + ' Sum', col + ' Average']
                                     , name='Total'))
        attr_tbls.append(temp)

    attr_tbls.insert(0, target_tbl)
    final_df = pd.concat(attr_tbls, axis=1)
    return (final_df)


