"""
MODEL FOR XX_atom_0
"""

print("[INFO] build and compiling model...")

zero = initializers.Zeros()
# kernel_initializer=zero
# kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)

model = tf.keras.Sequential()
model.add(layers.Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                beta_constraint=None, gamma_constraint=None))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                beta_constraint=None, gamma_constraint=None))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                beta_constraint=None, gamma_constraint=None))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))

rms = RMSprop(lr=0.0001)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#ada = Adagrad(lr=0.01, epsilon=None, decay=0.0)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

mse = 'mse'
#log = losses.mean_squared_logarithmic_error(y_train, y_val)

model.compile(optimizer=rms, # 'rmsprop
                    loss=mse,
                    metrics=['mae'])

model.output_shape
model.summary()
model.get_config()
model.get_weights()

print("[INFO] training model...")

history = model.fit(X_train, y_train,
            epochs=300, #200 for images 1.2, 2.2 and model_1.2 / 300 for images 1.3, 2.3 and model_1.3 (based on small_features_3)
            verbose=1,
            batch_size=1000,
            validation_data=(X_val, y_val))

print("[INFO] evaluating model...")

score = model.evaluate(X_test,
                        y_test
                        )

print(score)
