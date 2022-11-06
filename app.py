import streamlit as st
from Chatbot import *
from chatbot_demo import create_chat
from streamlit_chat import message
from streamlit_option_menu import option_menu
import pandas as pd


# Initialize the page configurations
st.set_page_config(
    page_title="Mikabot Project",
    page_icon=":broken_heart:",
    layout= "wide"
)

st.header("Mikabot- The Depressed and Sarcastic Bot")

# Create a navigation bar
col, _ = st.columns([2,1])
with col:
    selected = option_menu(
        menu_title=None,  # required
        options=["Home", "Chatbot"],  # required
        icons=["house", "robot"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"font-size": "30px"},
            "nav-link": {
                "font-size": "30px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#FF7361",
            },
            "nav-link-selected": {"background-color": "#FF4B4B"},
        },
    )

if selected == "Chatbot":
    st.markdown('---')
    _ , col_chat, _ = st.columns([1,1.5,1])
    with col_chat:
        with st.expander(label = "Disclaimers and explanations"):
            st.markdown('''<b> Mikabot is a depressed chatbot which will answer your question sarcastically and reluctantly </b>
            <ul>
            <li>Note that it can't understand the context of the conversation yet </li>
            <li>It will try to randomize the output based on the unknow input (names, etc) so you might get a weird answer </li>
            <li>Made as a joke. Please bear if the answer is offensive to you </li></ul>
            ''', unsafe_allow_html = True)
        create_chat()

if selected == "Home":
    st.markdown("---")
    st.markdown(""" <style>
                .big-font {
                    font-size:20px !important;
                }


                .fullScreenFrame > div {
                    display: flex;
                    justify-content: center;
                }
            </style> """, unsafe_allow_html=True)
    st.markdown("<h4>Using Seq2Seq Model to create a depressed and sarcastic chatbot</h4>", unsafe_allow_html= True)
    st.markdown("<h5> Introduction </h5>", unsafe_allow_html= True)
    st.markdown('''
    <p class = 'big-font'> 
    I started learning about machine learning recently back in university thanks to the <a href = "https://www.ntu.edu.sg/eee/student-life/mlda"> MLDA@EEE </a> club. One major study of machine learning I was 
    most interested in is in the domain of NLP (Natural Language Processing) and I fell in love ❤️ with it. Moving forward I would like to build up my
    skills to get a better job and after contemplating what to do in life, I believe I can try to improve my machine learning skills by creating my own personal project! \t </p>

    <p class = 'big-font'> Inspired by GPT3's <a href = "https://beta.openai.com/examples/default-marv-sarcastic-chat"> Marv the sarcastic chatbot </a>, I decided to create my own personal chatbot for fun. Now it is just a matter of
    deciding the personality of the bot! I would go for sarcastic as well but I would like the bot to be more "meme-able" and realistic. I talked to my friends
    for some inspirations and one of the say that I should get a virtual girlfriend or a robotic girlfriend. Because 
    I am single and she left a huge hole in my heart, therefore I decided with the robotic girlfriend idea.\t
    </p>
    ''', unsafe_allow_html= True)

    _, col_img, _ = st.columns([1,1.2,1])
    with col_img:
        st.image(image = "https://media.tenor.com/3inV_2EnxcsAAAAd/ouch-heartbroken.gif", caption= "My heart", use_column_width = True)

    st.markdown('''    
    <p class = 'big-font'> 
    She was smart, witty, sarcastic, and funny and I decided to incorporate her personality to the bot, thus the name Mikabot. But being all lovey-dovey is cringey and therefore I 
    also decided to incorporate my teacher's and my own personality and gave a result of this abomination of a bot
    </p> ''', unsafe_allow_html= True)
    
    st.markdown(''' <p class = 'big-font'>
    The seq2seq model also called the encoder-decoder model uses Long Short Term Memory- LSTM for text generation from the training corpus.The path to creating virtual girlfriend was not easy. I used Seq2Seq model in this case as I have a predefined output that I want the bot to say. While it is not
    perfect, but I think this bot is somewhat decent. I just require more training datas for the bot and it is hard to create your own dataset and some of the other dataset is not
    really usable for my case. But I managed to make some of the speech somewhat coherent.  </p>
    </p> ''', unsafe_allow_html= True)

    with st.expander(label = "Here is some of my conversation with my virtual gf.", expanded = False):
        st.markdown('''
        <li><em><strong>User:</strong></em> Hello </li>
        <li><em><strong>Bot:</strong></em> Hey </li>
        <li><em><strong>User:</strong></em> How are you </li>
        <li><em><strong>Bot:</strong></em> Dead inside. </li>
        <li><em><strong>User:</strong></em> I am good </li>
        <li><em><strong>Bot:</strong></em> Nein </li>
        <li><em><strong>User:</strong></em> Why are you dead inside </li>
        <li><em><strong>Bot:</strong></em> Default parametes </li>
        <li><em><strong>User:</strong></em> What is your name </li>
        <li><em><strong>Bot:</strong></em> Mikabot. Apparently my creator is salty about someone </li>
        ''', unsafe_allow_html= True)

    st.markdown("<h4> Some theory </h4>", unsafe_allow_html= True)
    st.markdown('''
    <p class = 'big-font'> 
    The seq2seq model also called the encoder-decoder model uses Long Short Term Memory- LSTM for text generation from the training data.
    But in my case I used GRU. GRU and LSTM are both a type of reccurent neural network. As the name suggest it 
    incorporates the memory to take any information from prior inputs to influence the current input and output! One difference between GRU and LSTM is that 
    GRU uses less training parameter and therefore uses less memory and executes faster than LSTM whereas LSTM is more accurate on a larger dataset.
    Since the training data is less than 4000 pairs of questions and answers, therefore I choose GRU for the implementation.
    </p>
    <p class = 'big-font'> 
    The seq2seq or encoder-decoder model predicts a word given in the user input and then each of the next words is predicted using the probability of likelihood of that word to occur.  The encoder output will become
    the decoder input. And we use a method is called teacher forcing.
    My analogy is that imagine a student and a teacher. And as the name suggest we try to make the student learn by providing some direct cheatsheet. Without a proper teacher the student
    might have the wrong concept and thus have a wrong prediction. 
    But during inference since there is no ground truth, it can lead the model to poor model performance and instability.
    Back to my analogy, imagine the student having to take a test without the cheatsheet, and since it has nothing to refer to it fails to do the test!
   <p class = 'big-font'> 
    ''', unsafe_allow_html= True)

    _, col_img2, _ = st.columns([1,1.5,1])
    with col_img2:
        st.image("https://s35691.pcdn.co/wp-content/uploads/2018/01/cheating-on-a-test-id181866634-180117.jpg", caption = "Teacher Forcing")
    st.markdown('''
    <p class = 'big-font'> 
    Back to the seq2seq modeling, the encoder outputs a sequence of vector that we can call 'context'.
    Based on this context, the decoder generates the output sequence, one word at a time while looking at the context and the previous word during each timestep. The image below might 
    give you a better understanding of how this works
    </p> 
    ''', unsafe_allow_html= True)
    _, col_img3, _ = st.columns([1,1.5,1])
    with col_img3:
        st.image("https://miro.medium.com/max/720/1*vNw0no_XkgianXvkxRGJ7w.png", caption = "Encoder and Decoder")

    st.markdown("<h3> The Data! </h3>", unsafe_allow_html= True)
    st.markdown('''
    <p class = 'big-font'> 
    I used a modified dataset from another <a href= "https://github.com/Machine-Learning-Tokyo/seq2seq_bot/tree/master/data">saracstic chatbot </a> project on github as a base and process it so that
    it has 2 columns of answer and question columns. Here I show some of the original data unedited from that project.
    </p>
    ''', unsafe_allow_html= True)

    df = pd.read_csv('dataset/Sample_data.csv', names=["Question", "Answer"])
    df = df.sample(100)
    st.dataframe(df, width = 1500, height= 600)

    st.markdown('---')

    st.markdown("<h3> The Codes </h3>", unsafe_allow_html= True)
    st.markdown("""<p class = 'big-font'> I wont explain much for the codes to train the model is already explained on the pytorch tutorial  <a href = "https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html"> 
    seq2seq modelling </a> and <a href = "https://pytorch.org/tutorials/beginner/chatbot_tutorial.html?highlight=chatbot"> generating chatbot </a>. I believe
    they can give a better explanation than I do. That's about it! Have fun chatting with the bot  </p>""", unsafe_allow_html= True)


    
