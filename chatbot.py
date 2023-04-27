import streamlit as st
import tensorflow as tf
import numpy as np
import argparse
import os
import data
import config

# Define global variables
sess = None
model = None
enc_vocab = None
dec_vocab = None
inv_dec_vocab = None

def init():
    # Load trained model and vocabularies
    global sess, model, enc_vocab, dec_vocab, inv_dec_vocab
    model, enc_vocab, dec_vocab, inv_dec_vocab = data.load_model()

def preprocess_input(text):
    # Preprocess user input
    return data.sentence2id(enc_vocab, text)

def generate_response(input_text):
    # Generate response using trained model
    global sess, model, enc_vocab, dec_vocab, inv_dec_vocab
    # Preprocess user input
    input_seq = preprocess_input(input_text)
    # Which bucket does it belong to?
    bucket_id = data._find_right_bucket(len(input_seq))
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, decoder_masks = data.get_batch([(input_seq, [])], 
                                                                    bucket_id,
                                                                    batch_size=1)
    # Get output logits for the sentence.
    _, _, output_logits = data.run_step(sess, model, encoder_inputs, decoder_inputs,
                                        decoder_masks, bucket_id, True)
    response = data._construct_response(output_logits, inv_dec_vocab)
    return response

def main():
    # Initialize model and vocabularies
    init()
    # Streamlit app UI
    st.sidebar.title("Chatbot")
    st.sidebar.write("Upload a text file with your message or enter text manually:")
    user_input = None
    uploaded_file = st.sidebar.file_uploader("", type=["txt"])
    if uploaded_file is not None:
        user_input = uploaded_file.read().decode("utf-8")
    col1, col2 = st.beta_columns(2)
    with col1:
        user_input = st.text_area("Enter text:", value="", height=200)
    with col2:
        st.write(" ")
    if st.button("Send"):
        # Generate response
        response = generate_response(user_input)
        st.write(response)

if __name__ == '__main__':
    main()
