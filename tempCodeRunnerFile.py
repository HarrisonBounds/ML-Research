        elif gan == "cwGAN":
            print("Using cwGAN ")\
            
            # preprocess data
            num_prep = make_pipeline(SimpleImputer(strategy='mean'),
                                    MinMaxScaler())
            cat_prep = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                    OneHotEncoder(handle_unknown='ignore'))
            
            prep = ColumnTransformer([
                ('num', num_prep, num_cols),
                ('cat', cat_prep, cat_cols)],
                remainder='drop')
            
            X_train_trans = prep.fit_transform(df_train)

            model = WGANGP(num_cols=num_cols, cat_cols=cat_cols, cat_dims=get_cat_dims(df_train, cat_cols), transformer=prep.named_transformers_['cat']['onehotencoder'], use_aux_classifier_loss=True)

            y_train = df_train[data["label_column"]]

            print("X_train shape: ", X_train.shape)
            print("y_train shape: ", y_train.shape)

            # print("X_train_trans: ", X_train_trans)
            # print("X_train_trans shape: ", X_train_trans.shape)
            # print("y_train shape: ", y_train.shape)
            # print("X_train_trans unique: ", X_train_trans[data["label_column"]].unique())
            # print("y_train unique: ", y_train.unique())

            model.fit(X_train.T,
                    y_train_encoded,
                    condition=True,
                    epochs=20,
                    batch_size=1000, 
                    netG_kwargs = {'hidden_layer_sizes': (128,64), 
                        'n_cross_layers': 1,
                        'cat_activation': 'gumbel_softmax',
                        'num_activation': 'none',
                        'condition_num_on_cat': True, 
                        'noise_dim': 30, 
                        'normal_noise': False,
                        'activation':  'leaky_relu',
                        'reduce_cat_dim': True,
                        'use_num_hidden_layer': True,
                        'layer_norm':False,},
                    netD_kwargs = {'hidden_layer_sizes': (128,64,32),
                        'n_cross_layers': 2,
                        'embedding_dims': 'auto',
                        'activation':  'leaky_relu',
                        'sigmoid_activation': False,
                        'noisy_num_cols': True,
                        'layer_norm':True,}
                )
            