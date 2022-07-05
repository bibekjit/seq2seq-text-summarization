from tensorflow.keras.layers import Dense, Bidirectional, LSTM, TimeDistributed
from tensorflow.keras.layers import Attention, Input, Embedding, Concatenate
from tensorflow.keras.models import Model


class Seq2Seq:
    def __init__(self,emb_dim=100,latent_dim=256,bi_layers=False,pretrained=False):
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.bi_layers = bi_layers
        self.pretrained = pretrained

        self.encoder_out = None
        self.encoder_in = None
        self.decoder_out = None
        self.decoder_in = None

    def build_encoder(self, max_input_len, x_vocab_size, trainable=False,
                      pretrained_weights=None, num_layers=2, dropout=0.2):
        self.encoder_in = Input(shape=(max_input_len,))
        if self.pretrained: weights = [pretrained_weights]
        else: weights = None
        emb_layer = Embedding(x_vocab_size, self.emb_dim, name="encoder_embedding",
                              trainable=trainable, weights=weights)

        emb = emb_layer(self.encoder_in)
        encoder = emb

        for i in range(num_layers):
            lstm_layer = LSTM(self.latent_dim, dropout=dropout, return_sequences=True,
                              return_state=True, recurrent_activation="sigmoid",name="encoder_"+str(i))
            if self.bi_layers:
                lstm_layer = Bidirectional(lstm_layer,name="encoder_"+str(i))
            encoder = lstm_layer(encoder)
        self.encoder_out = encoder
        if len(encoder) > 3:
            rnn_out, fwd_h, fwd_c, back_h, back_c = encoder
            state_h = Concatenate(name="encoder_state_h")([fwd_h, back_h])
            state_c = Concatenate(name="encoder_state_c")([fwd_c, back_c])
            self.encoder_out = [rnn_out, state_h, state_c]

    def build_decoder(self,y_vocab_size,pretrained_weights=None):
        self.decoder_in = Input(shape=(None,))

        if self.pretrained:
            weights = [pretrained_weights]
            trainable = False
        else:
            weights = None
            trainable = True
        emb_layer = Embedding(y_vocab_size, self.emb_dim, trainable=trainable,
                                  weights=weights, name="decoder_embedding")
        emb = emb_layer(self.decoder_in)
        latent_dim = self.latent_dim
        if self.bi_layers:
            latent_dim = self.latent_dim*2

        lstm_layer = LSTM(latent_dim, return_sequences=True, name="decoder_lstm",
                          return_state=True, recurrent_activation="sigmoid")

        lstm_out,_,_ = lstm_layer(emb,initial_state=self.encoder_out[1:])
        attn_out = Attention()([lstm_out,self.encoder_out[0]])
        lstm_out = Concatenate()([attn_out,lstm_out])
        dense = Dense(y_vocab_size,activation="softmax")
        self.decoder_out = dense(lstm_out)

    def stack_and_compile(self,optimizer="adam"):
        model = Model([self.encoder_in,self.decoder_in],self.decoder_out)
        model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer)
        return model

    @staticmethod
    def encoder_inference_model(model):
        en_in = model.input[0]
        layer_names = [x.name for x in model.layers]
        if max([len(x.output_shape) for x in model.layers]) == 5:
            en_out = [x.output[0] for x in model.layers if len(x.output_shape) == 5][-1]
            state_h = model.layers[layer_names.index("encoder_state_h")].output
            state_c = model.layers[layer_names.index("encoder_state_c")].output
            en_out = [en_out, state_h, state_c]
        else:
            en_out = [x.output for x in model.layers if "encoder" in x.name][-1]
        return Model(en_in, en_out)

    @staticmethod
    def decoder_inference_model(model):
        try:
            dec_in = Input(shape=(None,))
            max_input_len = model.input[0].shape[1]
            units = [x.output_shape[0][-1] for x in model.layers if "lstm" in x.name][-1]
            dec_in_h = Input(shape=(units,))
            dec_in_c = Input(shape=(units,))
            dec_hid_in = Input(shape=(max_input_len, units))
            dec_emb = [x for x in model.layers if "embedding" in x.name][-1]
            dec_layer = [x for x in model.layers if x.name == "decoder_lstm"][0]
            dec_out, h_inf, c_inf = dec_layer(dec_emb.output, initial_state=[dec_in_h,dec_in_c])
            attn = Attention()([dec_out, dec_hid_in])
            dec_out = Concatenate(axis=-1)([attn, dec_out])
            dec_out = model.layers[-1](dec_out)
            Model([dec_in, dec_hid_in, dec_in_h, dec_in_c], [dec_out, h_inf, c_inf])
        except:
            dec_in = model.input[1]
            max_input_len = model.input[0].shape[1]
            units = [x.output_shape[0][-1] for x in model.layers if "lstm" in x.name][-1]
            dec_in_h = Input(shape=(units,))
            dec_in_c = Input(shape=(units,))
            dec_hid_in = Input(shape=(max_input_len, units))
            dec_emb = [x for x in model.layers if "embedding" in x.name][-1]
            dec_layer = [x for x in model.layers if x.name == "decoder_lstm"][0]
            dec_out, h_inf, c_inf = dec_layer(dec_emb.output, initial_state=[dec_in_h, dec_in_c])
            attn = Attention()([dec_out, dec_hid_in])
            dec_out = Concatenate()([attn, dec_out])
            dec_out = model.layers[-1](dec_out)
            return Model([dec_in, dec_hid_in, dec_in_h, dec_in_c], [dec_out, h_inf, c_inf])


































