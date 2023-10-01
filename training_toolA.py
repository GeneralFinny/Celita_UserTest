import numpy as np
import openai
from googletrans import Translator
import tensorflow as tf
import pickle
from tensorflow.keras import layers , activations , models , preprocessing
from tensorflow.keras import preprocessing , utils
import os
import yaml
import json
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import pickle
import re


def processTweet(chat):
    if isinstance(chat, str):  # Check if chat is a string
        chat = chat.lower()
        chat = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', chat)
        chat = re.sub('@[^\s]+', '', chat)
        chat = re.sub('[\s]+', ' ', chat)
        chat = re.sub(r'#([^\s]+)', r'\1', chat)
        chat = re.sub(r'[\.!:\?\-\'\"\\/]', r'', chat)
        chat = chat.strip('\'"')
        return chat
    else:
        return chat  # Return the input as is if it's not a string


def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    return pattern.sub(r"\1", s)
def getFeatureVector(chat):
    if isinstance(chat, str):  # Check if chat is a string
        chat = processTweet(chat)
        featureVector = []
        # Split tweet into words
        words = chat.split()
        for w in words:
            # Replace two or more with two occurrences
            w = replaceTwoOrMore(w)
            # Strip punctuation
            w = w.strip('\'"?,.')
            # Check if the word starts with an alphabet
            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
            # Ignore if it is a stop word
            if val is None:
                continue
            else:
                featureVector.append(w.lower())
        return " ".join(list(featureVector))
    else:
        return chat  # Return the input as is if it's not a string
def emb_mat(nb_words):
    EMBEDDING_FILE = "Embedder/glove.6B.100d.txt"
    

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding="utf8"))

    all_embs = [embeddings_index[word] for word in word_index if word in embeddings_index]
    all_embs = np.asarray(all_embs, dtype='float32')

    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words + 1, embed_size))
    
    for word, i in word_index.items():
        if (i >= max_features) or i == nb_words:
            continue
        embedding_vector = embeddings_index.get(word)  # here we will get embedding for each word from GloVe
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def tokenized_data(questions,answers,VOCAB_SIZE,tokenizer):
    # encoder_input_data
    import numpy as np
    tokenized_questions = tokenizer.texts_to_sequences( questions )
    maxlen_questions = max( [ len(x) for x in tokenized_questions ] )
    padded_questions = preprocessing.sequence.pad_sequences( tokenized_questions , maxlen=maxlen , padding='post' )
    encoder_input_data = np.array( padded_questions )
    #print( encoder_input_data.shape , maxlen_questions )

    # decoder_input_data
    tokenized_answers = tokenizer.texts_to_sequences( answers )
    maxlen_answers = max( [ len(x) for x in tokenized_answers ] )
    padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen , padding='post' )
    decoder_input_data = np.array( padded_answers )
    #print( decoder_input_data.shape , maxlen_answers )

    # decoder_output_data
    tokenized_answers = tokenizer.texts_to_sequences( answers )
    for i in range(len(tokenized_answers)) :
        tokenized_answers[i] = tokenized_answers[i][1:] # remove <start> take rest
    padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen , padding='post' )
    onehot_answers = utils.to_categorical( padded_answers , VOCAB_SIZE)
    decoder_output_data = np.array( onehot_answers )
    #print( decoder_output_data.shape )
    
    return [encoder_input_data,decoder_input_data,decoder_output_data,maxlen_answers]
def enable_gpu():
    #config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
    #sess = tf.compat.v1.Session(config=config) 
    #tf.compat.v1.keras.backend.set_session(
    #    sess
    #)
    #gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.333)
    #session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

def get_model(nb_words,embed_size,embedding_matrix):
    encoder_inputs = tf.keras.layers.Input(shape=(None,))
    encoder_embedding = tf.keras.layers.Embedding(nb_words + 1, embed_size, mask_zero=True, weights=[embedding_matrix])(
        encoder_inputs)
    encoder_bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(encoder_embedding)
    state_h = layers.Concatenate()([forward_h, backward_h])
    state_c = layers.Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    decoder_inputs = tf.keras.layers.Input(shape=( None ,  ))
    decoder_embedding = tf.keras.layers.Embedding( nb_words+1, embed_size , mask_zero=True,weights=[embedding_matrix]) (decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
    decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )

    decoder_dense = tf.keras.layers.Dense( nb_words+1 , activation=tf.keras.activations.softmax ) 
    output = decoder_dense ( decoder_outputs )
    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=( 200 ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 200 ,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h_d, state_c_d = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h_d, state_c_d]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
    return model,encoder_model,encoder_inputs,encoder_states,decoder_lstm,decoder_embedding,decoder_dense,decoder_inputs,decoder_outputs,decoder_model,nb_words


def translate(input_text, enc_model, dec_model, tokenizer):
    def str_to_tokens(sentence: str):
        words = sentence.lower().split()
        tokens_list = list()
        for word in words:
            tokens_list.append(tokenizer.word_index[word])
        return preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen, padding='post')

    states_values = enc_model.predict(str_to_tokens(input_text))
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition:
        dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                decoded_translation += ' {}'.format(word)
                sampled_word = word
        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]
    return " ".join(decoded_translation.strip().split(" ")[:-1])

def prepare_data(questions, answers):
        answers = pd.DataFrame(answers, columns=["Ans"])
        questions = pd.DataFrame(questions, columns=["Question"])
        questions["TokQues"] = questions["Question"].apply(getFeatureVector)

        # Filter out unwanted elements from questions and answers
        valid_indices = []
        for i in range(len(answers)):
            if isinstance(answers["Ans"][i], str):
                valid_indices.append(i)
        questions = questions.iloc[valid_indices]
        answers = answers.iloc[valid_indices]

        answers = list(answers["Ans"])
        questions = list(questions["TokQues"])

        answers_with_tags = list()
        for i in range( len( answers ) ):
            if type( answers[i] ) == str:
                answers_with_tags.append( answers[i] )
            else:
                print(questions[i])
                print(answers[i])
                print(type(answers[i]))
                questions.pop(i)

        answers = list()
        for i in range( len( answers_with_tags ) ) :
            answers.append( '<START> ' + answers_with_tags[i] + ' <END>' )
        
        
        tokenizer = preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(questions+answers)

        with open('tokenizer.pkl', 'wb') as tokenizer_file:
            pickle.dump(tokenizer, tokenizer_file)


        word_index = tokenizer.word_index
        nb_words = min(max_features, len(word_index))

        #embedding_matrix=emb_mat(nb_words)[0]
        #emb_vec=emb_mat(nb_words)[1]

        VOCAB_SIZE = len( tokenizer.word_index )+1
        
        
        tok_out=tokenized_data(questions,answers,VOCAB_SIZE,tokenizer)
        encoder_input_data=tok_out[0]
        decoder_input_data=tok_out[1]
        decoder_output_data=tok_out[2]
        maxlen_answers=tok_out[3]
        
        return [encoder_input_data,decoder_input_data,decoder_output_data,maxlen_answers,nb_words,word_index,tokenizer]

def make_inference_models():
            
            encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
            
            decoder_state_input_h = tf.keras.layers.Input(shape=( 200 ,))
            decoder_state_input_c = tf.keras.layers.Input(shape=( 200 ,))
            
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            
            decoder_outputs, state_h, state_c = decoder_lstm(
                decoder_embedding , initial_state=decoder_states_inputs)
            decoder_states = [state_h, state_c]
            decoder_outputs = decoder_dense(decoder_outputs)
            decoder_model = tf.keras.models.Model(
                [decoder_inputs] + decoder_states_inputs,
                [decoder_outputs] + decoder_states)
            
            return encoder_model , decoder_model

# define the checkpoint


print("Choose an option:")
print("1. Train")
print("2. Translate")
choice = input("Enter your choice (1 or 2): ")

if choice == "1":

    filipino_docs = open('Datasets/filipino_experiment.txt', encoding='utf-8').read().split("\n")
    cebuano_docs = open('Datasets/cebuano_experiment.txt', encoding='utf-8').read().split("\n")

    questions_for_token = list()
    answers_for_token = list()

    for filipino_con, cebuano_con in zip(filipino_docs, cebuano_docs):
        if filipino_con and cebuano_con:
            filipino_con = filipino_con.strip()
            cebuano_con = cebuano_con.strip()
            questions_for_token.append(filipino_con)  # Filipino sentences
            answers_for_token.append(cebuano_con)    # Cebuano translations

    embed_size = 100  # how big is each word vector
    max_features = 6000
    maxlen = 50
   

   
    

    filepath = "Models/model1.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
   
   
    enable_gpu()
    Prepared_data=prepare_data(questions_for_token,answers_for_token)
    print("Preparation done")
    encoder_input_data=Prepared_data[0]
    print("Encoder done")
    decoder_input_data=Prepared_data[1]
    decoder_output_data=Prepared_data[2]
    print("Decoder done")
    maxlen_answers=Prepared_data[3]
    nb_words=Prepared_data[4]
    word_index=Prepared_data[5]
    tokenizer=Prepared_data[6]
    print(encoder_input_data.shape, decoder_input_data.shape, decoder_output_data.shape)

    embedding_matrix=emb_mat(nb_words)
    #model=get_model(nb_words,embed_size,embedding_matrix)[0]
    #encoder_model=get_model(nb_words,embed_size,embedding_matrix)[1]
    #encoder_inputs=get_model(nb_words,embed_size,embedding_matrix)[2]
    #encoder_states=get_model(nb_words,embed_size,embedding_matrix)[3]
    #decoder_lstm=get_model(nb_words,embed_size,embedding_matrix)[4]
    #decoder_embedding=get_model(nb_words,embed_size,embedding_matrix)[5]
    #decoder_dense=get_model(nb_words,embed_size,embedding_matrix)[6]
    #decoder_inputs=get_model(nb_words,embed_size,embedding_matrix)[7]
    #decoder_outputs=get_model(nb_words,embed_size,embedding_matrix)[8]
    #decoder_model=get_model(nb_words,embed_size,embedding_matrix)[9]

    encoder_inputs = tf.keras.layers.Input(shape=( None , ))
    encoder_embedding = tf.keras.layers.Embedding( nb_words+1, embed_size , mask_zero=True, weights=[embedding_matrix]) (encoder_inputs)
    encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
    encoder_states = [ state_h , state_c ]

    decoder_inputs = tf.keras.layers.Input(shape=( None ,  ))
    decoder_embedding = tf.keras.layers.Embedding( nb_words+1, embed_size , mask_zero=True,weights=[embedding_matrix]) (decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
    decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )

    decoder_dense = tf.keras.layers.Dense( nb_words+1 , activation=tf.keras.activations.softmax ) 
    output = decoder_dense ( decoder_outputs )

    model = models.Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=10, epochs=30, callbacks=callbacks_list)
    model.save_weights("model_weights.h5")
    print("Model weights saved to 'model_weights.h5'")
    import numpy as np

    # Path to the GloVe word vectors file
    glove_file_path = 'Embedder/glove.6B.100d.txt'

    # Load the GloVe word vectors into a dictionary
    def load_glove_vectors(file_path):
        embeddings_index = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    embeddings_index = load_glove_vectors(glove_file_path)

    # Define the vocabulary size and embedding dimension
    vocab_size = 6000  # Adjust this value based on your dataset
    embedding_dim = 100  # This should match the dimension of your GloVe embeddings (100 in this case)

    # Initialize the embedding matrix with random values
    embedding_matrix = np.random.rand(vocab_size + 1, embedding_dim)

    # Fill the embedding matrix with GloVe embeddings for words in your vocabulary
    for word, i in tokenizer.word_index.items():
        if i <= vocab_size and word in embeddings_index:
            embedding_vector = embeddings_index[word]
            embedding_matrix[i] = embedding_vector

    # Save the embedding matrix to a file
    np.save('embedding_matrix.npy', embedding_matrix)

    print("Embedding matrix saved as 'embedding_matrix.npy'")


    # Path to the GloVe text file
    glove_file = 'Embedder/glove.6B.100d.txt'

    # Load GloVe embeddings into a dictionary
    embeddings_index = {}
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Define your vocabulary and embedding dimension
        
        def str_to_tokens( sentence : str ):
            words = sentence.lower().split()
            tokens_list = list()
            for word in words:
                tokens_list.append( tokenizer.word_index[ word ] ) 
            return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxlen , padding='post')
        enc_model , dec_model = make_inference_models()
        for _ in range(100):
            user_input = input("Enter: ")
            try:
                states_values = enc_model.predict(str_to_tokens(user_input))

                
                empty_target_seq = np.zeros( ( 1 , 1 ) )
                empty_target_seq[0, 0] = tokenizer.word_index['start']
                stop_condition = False
                decoded_translation = ''
                while not stop_condition :
                    dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
                    sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
                    sampled_word = None
                    for word , index in tokenizer.word_index.items() :
                        if sampled_word_index == index :
                            decoded_translation += ' {}'.format( word )
                            sampled_word = word
                    
                    if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
                        stop_condition = True
                        
                    empty_target_seq = np.zeros( ( 1 , 1 ) )  
                    empty_target_seq[ 0 , 0 ] = sampled_word_index
                    states_values = [ h , c ] 

                print( " ".join(decoded_translation.strip().split(" ")[:-1]) )
                
            except:

                from googletrans import Translator
                translator = Translator()
                translated_text = translator.translate(user_input, dest = 'ceb')
                print(translated_text.text)





elif choice == "2":
    # Load the pre-trained model and tokenizer
    model = tf.keras.models.load_model("Models/model1.h5")
    with open('tokenizer.pkl', 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    



    # Load the encoder and decoder models
    enc_model, dec_model = make_inference_models()



    # Define functions for making inferences and performing translations
    def str_to_tokens(sentence: str):
        words = sentence.lower().split()
        tokens_list = list()
        for word in words:
            tokens_list.append(tokenizer.word_index[word])
        return preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen, padding='post')

    def translate(input_text, enc_model, dec_model, tokenizer):
        states_values = enc_model.predict(str_to_tokens(input_text))
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = tokenizer.word_index['start']
        stop_condition = False
        decoded_translation = ''
        while not stop_condition:
            dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = None
            for word, index in tokenizer.word_index.items():
                if sampled_word_index == index:
                    decoded_translation += ' {}'.format(word)
                    sampled_word = word
            if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
                stop_condition = True
            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]
        return " ".join(decoded_translation.strip().split(" ")[:-1])

    # Load the encoder and decoder models

enc_model, dec_model = make_inference_models(encoder_inputs)

while True:
    user_input = input("Enter a sentence in Filipino: ")
    if user_input.lower() == 'exit':
        break
    translated_text = translate(user_input, enc_model, dec_model, tokenizer)
    print("Translated Cebuano: " + translated_text)
    tf.keras.backend.clear_session()

   