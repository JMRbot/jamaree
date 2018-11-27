import  pickle
from keras.models import model_from_json
import numpy as np
from numpy import array
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model
from keras.preprocessing import sequence
from keras.layers import Dense
import deepcut

def define_models(n_input, n_output, n_units):
    # encoder model
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # decoder model
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # รูปแบบตัวเข้ารหัสที่ใช้เมื่อทำนายลำดับแหล่งข้อมูลใหม่
    encoder_model = Model(encoder_inputs, encoder_states)
    # รูปแบบตัวถอดรหัสที่ใช้เมื่อทำนายลำดับแหล่งข้อมูลใหม่
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model

# one hot encode
def one_hot_encode(X, max_int):
    Xenc = list()
    for seq in X:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Xenc.append(pattern)

    return Xenc

# invert encoding
def invert(seq):
    strings = list()
    for pattern in seq:
        string = int_to_word_input[np.argmax(pattern)]
        if(string!="padd"):
            strings.append(string)
    return strings

# สร้าง target ที่เกิดขึ้นตามลำดับ
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    # encode
    state = infenc.predict(source)
    # start of sequence input
    target_seq = array(one_hot_encode(array([[word_to_int_input["<go>"]]]),encoded_length))
    # collect predictions
    output = list()
    for t in range(n_steps):
        # ทำนายคำต่อไป
        yhat, h, c = infdec.predict([target_seq] + state)
        # store prediction
        output.append(yhat[0,0,:])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat
    return array(output)

# load integer encoding dict
classifier_f = open("word_to_int.pickle", "rb")
word_to_int_input = pickle.load(classifier_f)
classifier_f.close()
classifier_f = open("int_to_word.pickle", "rb")
int_to_word_input = pickle.load(classifier_f)
classifier_f.close()

# define model
encoded_length=len(word_to_int_input)
train, infenc, infdec = define_models(encoded_length, encoded_length, 256)

# ค่าน้ำหนัก load weights
infenc.load_weights("model_encode.h5")
infdec.load_weights("model_decode.h5")

# chatbot เริ่ม ทำงาน
def Chatbot(input_data):
    checkword=[]
    print('user_w',input_data)
    input_data = deepcut.tokenize(input_data) ##ตัดคำที่รับเข้ามาด้วย deepcut    ##['สวัสดี']
    if input_data[0] != "สวัสดี":
        for check in input_data:
            if check in word_to_int_input:
                #print('check',check)
                checkword.append(check)
##        print('w',checkword)
        checkword=[word_to_int_input[word] for word in checkword]
        checkword=np.array([checkword])
        checkword = sequence.pad_sequences(checkword, maxlen=40,padding='post')
        checkword=one_hot_encode(checkword,encoded_length)
        checkword=array(checkword)
     ##แก้   target = predict_sequence(infenc, infdec, checkword, 24, encoded_length)
        target = predict_sequence(infenc, infdec, checkword,10, encoded_length)  
        #print('target ',target)
        #print('bot',invert(target))
        words=""
        for word in invert(target):
            words = words+word
        input_data = ""
        return words
    else:
        words = "สวัสดี"
        return words
        
##cc = Chatbot("บัตรนิสิตหาย")
##print("cc", cc)
        
