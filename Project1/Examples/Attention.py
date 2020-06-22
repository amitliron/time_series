
# -- 1 ----
model.add(LSTM(200, return_sequences=False, activation='softmax'))

# -- 2 ----
model.add(LSTM(200, return_sequences=False))
model.add(Activation('softmax')) #this guy here