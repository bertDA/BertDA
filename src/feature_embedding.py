import pandas as pd
import csv
import argparse



def generate_seq1Tag(df):
    df['Seq1Tag'] = df['Role'].apply(lambda x: x[0])
    return df

def generate_seq2Tag(df, sess_uniq):

    for eachSess in sess_uniq:
        eachSess_df = df[df['Session_id'] == eachSess]
        pre_role = ''
        begin = 'B'
        for idx, row in eachSess_df.iterrows():
            cur_role = row['Seq1Tag']
            cur_utter = row['Utterances']

            if begin == 'B':
                df.loc[idx, 'Seq2Tag'] = begin + cur_role
                pre_role = cur_role
                df.loc[idx, 'prevUtterances'] = cur_utter
                pre_utter = cur_utter
                begin = None
            else:

                if (eachSess_df.index[-1] == idx):
                    df.loc[idx, 'Seq2Tag'] = pre_role + cur_role + 'E'
                    df.loc[idx, 'prevUtterances'] = pre_utter
                else:
                    df.loc[idx, 'Seq2Tag'] = pre_role + cur_role
                    df.loc[idx, 'prevUtterances'] = pre_utter
                    pre_role = cur_role
                    pre_utter = cur_utter
    return df

def generate_seq3Tag(df, sess_uniq):

    for eachSess in sess_uniq:
        temp_df = df[df['Session_id'] == eachSess]

        for idx, row in temp_df.iterrows():

            if (temp_df.index[0] == idx):
                df.loc[idx, 'Seq3Tag'] = df.loc[idx, 'Seq2Tag']
            elif (temp_df.index[-1] == idx):
                df.loc[idx, 'Seq3Tag'] = df.loc[idx - 1, 'Seq2Tag'] + df.loc[idx, 'Seq1Tag'] + 'E'
            else:
                df.loc[idx, 'Seq3Tag'] = df.loc[idx - 1, 'Seq2Tag'] + df.loc[idx, 'Seq1Tag']
    return df

def generate_new_utterances(df):
    cls = "[CLS]"
    sep = "[SEP]"
    s = ' '

    df['new_Utter'] = cls + s + df['Session_id'].astype(str) + s + df['Role'] + s + df['prevUtterances'] + s + sep + s \
            + cls + s + df['Seq2Tag'] + s + df['Seq3Tag'] + s + df['Utterances'] + s + sep

    # df['new_Utter'] = cls + s + df['Session_id'].astype(str) + s + df['Role'] + s + df['Seq2Tag'] + \
    #                   s + df['Seq3Tag'] + s + sep + s + cls + s + df['Utterances'] + s + sep

    return df

def remove_invaild_symbol(df):

    for i in range (5):
        df['new_Utter'] = df['new_Utter'].apply(lambda x: x.replace('\n', ' '))
        df['new_Utter'] = df['new_Utter'].apply(lambda x: x.replace('\t', ' '))
    return df


def write_csv(df, path, filename, colNames):
    df[['new_Utter', colNames]].to_csv(path + filename + '.csv', header=None, index=False)

def generate_tsv(path, filename):
    csv.writer(open(path + filename + '.tsv', 'w+'), delimiter='\t').writerows(
        csv.reader(open(path + filename + '.csv')))


def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default='../data/feedback/',
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument("--train", type=str, default='', help="process train csv")
    parser.add_argument("--test", type=str, default='', help="process test csv")
    parser.add_argument("--wFlag", type=str, default='', help="process test csv")
    parser.add_argument("--levelFlag", type=str, default='firstlevel', help="assign the label name")

    args = parser.parse_args()

    return args

def process_tsv_files(path, filename, type, colNames):

    pathfilename = path + filename + '.csv'

    df = pd.read_csv(pathfilename)
    sess_uniq = df['Session_id'].unique()
    df = generate_seq1Tag(df)
    df = generate_seq2Tag(df, sess_uniq)
    df = generate_seq3Tag(df, sess_uniq)
    df = generate_new_utterances(df)
    df = remove_invaild_symbol(df)

    if type == 'train':
        df = duplicate_imbalance_dataset(df, colNames)

    write_filename = filename + '_' + colNames + '_Wr'
    write_csv(df, path, write_filename, colNames)
    generate_tsv(path, write_filename)


def sample_dup(df, num, label, colNames):
    label_C = df[df[colNames] == label]
    label_C = label_C.sample(n = num, replace=True)
    return label_C

def duplicate_imbalance_dataset(df, colNames):

    lu_df = df.groupby([colNames]).count().reset_index()[[colNames, 'new_Utter']]

    for _, row in lu_df.iterrows():
        label_c = sample_dup(df, max(lu_df['new_Utter'].values) - row['new_Utter'], row[colNames], colNames)
        df = pd.concat([label_c, df])
    df.reset_index(inplace=True)
    df = df.drop(['index'], axis=1)
    return df


def main():

    args = get_args()
    print(args)
    path = args.data_dir

    if args.levelFlag == 'firstlevel':
        colNames = 'First_Label'
    elif args.levelFlag == 'secondlevel':
        colNames = 'Second_Label'

    if (args.wFlag == 'train'):
        process_tsv_files(path, args.train, args.wFlag, colNames)

    elif (args.wFlag == 'test'):
        process_tsv_files(path, args.test, args.wFlag, colNames)
    else:
        raise AssertionError('please enter --wFlag argument')

if __name__ == "__main__":
    main()
