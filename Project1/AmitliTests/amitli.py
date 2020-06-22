# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import numpy as np

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        seq_x = np.insert(seq_x, 0, i)
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def main():
    # define input sequence

    raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps+1, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=200, verbose=0)
    # demonstrate prediction
    #x_input = array([70, 80, 90])
    x_input = array([2, 70, 80, 90])
    x_input = x_input.reshape((1, n_steps+1, n_features))
    yhat = model.predict(x_input, verbose=0)
    print("result: ", yhat)


def test_ks():
    a   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    gt0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5]
    gt1 = [0.2, 0.2, 0.2, 0.2, 0.2, 0.7, 0.7, 0.7, 0.7, 0.7]
    #gt0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5]
    #gt1 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.7, 0.5, 0.5]

    from scipy import stats
    ks_stat, p_value = stats.ks_2samp(gt0, gt1)
    print("p_value = ", p_value)
    print("ks_stat = ", ks_stat)
    None

if __name__ == "__main__":
    test_ks()
    main()