import pandas as pd

def get_data(split = True):
    features = list()
    with open("data/features.txt") as f:
        features = [line.split()[1] for line in f.readlines()]

    # print("Number of Features: {}".format(len(features)))

    # read in
    x_train_path = 'data/train/X_train.txt'
    X_train = pd.read_csv(x_train_path, sep=r'\s+', header=None) 
    X_train.columns = features

    y_train_path = 'data/train/y_train.txt'
    y_train = pd.read_csv(y_train_path, names=['Activity'])

    # Map activity labels to their names
    y_train['Activity'] = y_train['Activity'].map({
        1: 'Walking',
        2: 'Walking_Upstair',
        3: 'Walking_Downstair',
        4: 'Sitting',
        5: 'Standing',
        6: 'Laying'
    })

    # Putting all columns in a single Dataframe

    train = X_train
    train['Activity'] = y_train
    # train.sample()

    # read in
    x_test_path = 'data/test/X_test.txt'
    X_test = pd.read_csv(x_test_path, sep=r'\s+', header=None)
    X_test.columns = features


    y_test_path = 'data/test/y_test.txt'
    y_test = pd.read_csv(y_test_path, names=['Activity'])

    # Map activity labels to their names
    y_test['Activity'] = y_test['Activity'].map({
        1: 'Walking',
        2: 'Walking_Upstair',
        3: 'Walking_Downstair',
        4: 'Sitting',
        5: 'Standing',
        6: 'Laying'
    })


    test = X_test
    test['Activity'] = y_test
    # test.sample()

    columns = train.columns
    # Removing (), - and , from column names
    columns = columns.str.replace('[()]', '', regex=True)
    columns = columns.str.replace('[-]', '', regex=True)
    columns = columns.str.replace('[,]', '', regex=True)

    # Assign cleaned column names back to train and test
    train.columns = columns
    test.columns = columns
    
    if split:
        return train, test
    else:
        # Combine train and test data
        combined = pd.concat([train, test], ignore_index=True)
        return combined
    



