import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pdb, ipdb

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
        modeling_df.drop([question_number], axis=1), modeling_df[question_number], test_size=test_size, random_state=42)

    return (X_train, X_test, y_train, y_test)

def get_cnt_pct(col):
    cnts = col.value_counts()
    pcts = np.round(col.value_counts(normalize=True)* 100, 2 )
    cnts_pcts = pd.concat([cnts, pcts], keys= ['Counts', "Percentage"], axis=1)
    return(cnts_pcts)

def preprocess_target(target, percentage_cutoff=5):
    """
    Preprocess target by removing nulls and combining low percentage targets into "Other group"

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
        target = target.replace(small_pcts.index.tolist(), 'Other')
        new_pcts = get_cnt_pct(target)
        print('\nNew target percentages: ', new_pcts)

    repl_dict = {sorted_vals.index[x]: x for x in range(sorted_vals.shape[0])}
    target = target2.replace(repl_dict)

    return (target, repl_dict)

