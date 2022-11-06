from Chatbot import *
from streamlit_chat import message
import streamlit as st

def create_chat():

    # Initialize the chatbot
    chatbot = Chatbot()
    chatbot.name = "Mikabot"

    ### Create the history settings ###
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    ### Text generation
    def get_text():
        input_text = st.text_input(label = "Chat with the bot", key="input", placeholder= "Your input")
        return input_text 

    user_input = get_text()
    if user_input:
        output = chatbot.reply(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    ##### Container Settings  #######

    placeholder = st.empty()

    with placeholder.container():
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed=123)
