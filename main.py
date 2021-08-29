import pandas as pd
from data_processing.preprocessing import Preprocessing
from data_processing.window_generator import WindowGenerator
from smart_billing import SmartBilling


def read_dataset():
    return pd.read_csv("./dataset/gpc_data.csv")


def run():
    df = read_dataset()

    # Remove date columns
    df.pop('dd_session_date')
    df.pop('usage_start_time')

    preprocessing = Preprocessing(df)

    train_df, val_df, test_df = preprocessing.split_and_normalize()

    train_window = WindowGenerator(input_width=100, label_width=100, shift=10,
                                   train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['total'])

    smart_billing = SmartBilling(train_window, model='dense')
    smart_billing.train(epoch=500)
    smart_billing.plot(input_width=10, label_width=10, shift=5)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()
