import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Preprocessing:

    def __init__(self, dataframe):
        self.n = len(dataframe)
        self.dataframe = dataframe.fillna(dataframe.mean())

    def split(self):
        train_df = self.dataframe[0:int(self.n * 0.7)]
        val_df = self.dataframe[int(self.n * 0.7):int(self.n * 0.9)]
        test_df = self.dataframe[int(self.n * 0.9):]

        return train_df, val_df, test_df

    def split_and_normalize(self):

        train_df = self.dataframe[0:int(self.n * 0.7)]
        val_df = self.dataframe[int(self.n * 0.7):int(self.n * 0.9)]
        test_df = self.dataframe[int(self.n * 0.9):]

        train_mean = train_df.mean()
        train_std = train_df.std()

        self.df_std = (self.dataframe - train_mean) / train_std

        train_df = (train_df - train_mean) / train_std
        val_df = (val_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std

        return train_df, val_df, test_df

    def plot_box(self):
        assert not self.df_std.empty

        df_std = self.df_std.melt(var_name='Column', value_name='Normalized')
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
        _ = ax.set_xticklabels(self.dataframe.keys(), rotation=90)
        plt.show()
