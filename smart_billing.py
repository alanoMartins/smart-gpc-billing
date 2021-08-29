from models.single_steps import MLP as MLP_single, LSTM as LSTM_single, Baseline as Baseline_single
# from models.multiple_steps import MLP as MLP_mult, LSTM as LSTM_mult, Baseline as Baseline_mult
import tensorflow as tf
from data_processing.window_generator import WindowGenerator


class SmartBilling:

    def __init__(self, window, model='lstm', multiple_step=False):
        self.window = window
        if multiple_step:
            pass
            # if model == 'dense':
            #     self.model = MLP_mult()
            # elif model == 'lstm':
            #     self.model = LSTM_mult()
            # else:
            #     self.model = Baseline_mult()
        else:
            if model == 'dense':
                self.model = MLP_single()
            elif model == 'lstm':
                self.model = LSTM_single()
            else:
                self.model = Baseline_single()

    def train(self, epoch):

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='min')
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir="./tensorboard")

        self.model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam(),
                           metrics=[tf.metrics.MeanAbsoluteError()])

        return self.model.fit(self.window.train, epochs=epoch, validation_data=self.window.val,
                              callbacks=[early_stopping, tensorboard])
    
    def plot(self, input_width, label_width, shift):
        wide_window = WindowGenerator(
            input_width=input_width, label_width=label_width, shift=shift,
            train_df=self.window.train_df, test_df=self.window.test_df, val_df=self.window.val_df,
            label_columns=['total'])
        wide_window.plot(self.model)
