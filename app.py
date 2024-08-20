# Author: Muhammad Hassaan Bashir
# Dated: 12-08-2024
# TextSummarizer and Email Agents using llama Index Based

# Importing required modules
# import chainlit as cl
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
# from llama_index.llms.huggingface import HuggingFaceLLM
# from transformers import HfEngine
# from llama_index.core import PromptTemplate
# from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.anthropic import Anthropic
import nest_asyncio
import logging
import requests
import torch
from llama_index.core import ServiceContext
from llama_index.core.response_synthesizers import TreeSummarize
# from huggingface_hub import login
import yaml
import os
import streamlit as st
import random
import time

os.environ["ANTHROPIC_API_KEY"] = st.secrets["anthropic_api_key"]
# print (os.environ['ANTHROPIC_API_KEY'])
# os.environ["ANTHROPIC_API_KEY"] = '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'

# Loading configs from yml file
# with open('./development.yml', 'r') as f:
#     config = yaml.load(f, Loader=yaml.SafeLoader)


# HF_TOKEN = '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'
# login(HF_TOKEN)

# This will wrap the default prompts that are internal to llama-index
# taken from https://huggingface.co/Writer/camel-5b-hf
# query_wrapper_prompt = PromptTemplate(
#     "Below is an instruction that describes a task. "
#     "Write a response that appropriately completes the request.\n\n"
#     "### Instruction:\n{query_str}\n\n### Response:"
# )


nest_asyncio.apply()

# Create and configure logger
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
# Creating an object
logger = logging.getLogger()
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

# llm = HuggingFaceLLM(
#     context_window=2048,
#     max_new_tokens=256,
#     generate_kwargs={"temperature": 0.25, "do_sample": False},
#     query_wrapper_prompt=query_wrapper_prompt,
#     # tokenizer_name="Writer/camel-5b-hf",
#     # model_name="Writer/camel-5b-hf",
#     tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
#     model_name="StabilityAI/stablelm-tuned-alpha-3b",
#     device_map="auto",
#     tokenizer_kwargs={"max_length": 2048},
    
#     # uncomment this if using CUDA to reduce memory usage
#     # model_kwargs={"torch_dtype": torch.float16}
# )

# Settings.chunk_size = 512


# Initializing llm
# llm = HfEngine(model="meta-llama/Meta-Llama-3-70B-Instruct")
# llm = HuggingFaceInferenceAPI(model_name="HuggingFaceH4/zephyr-7b-alpha", api_key=HF_TOKEN)
# llm = Ollama(model='stablelm-zephyr', request_timeout=300.0)
# llm = Ollama(model='tinyllama', request_timeout=300.0)
# # model_name = 'claude-3-sonnet-20240229'
model_name = 'claude-3-opus-20240229'
llm = Anthropic(model=model_name)

# Setting Ollama llm as llm for llama index 
Settings.llm = llm


# Tools for AI Gents
# 1. Text Summarizer Tool
def textSummarizer (user_input: str) -> str:
    service_context = ServiceContext.from_defaults(llm=llm)
    summarizer = TreeSummarize(service_context=service_context,verbose=True)
    response =  summarizer.get_response("what is all about?", [user_input])
    return response
    
# 2. Email Sending Tool
def emailSender (summary:str, recipient_email: str, recipient_name: str=None) -> str:
    # if recipient_email == '':
    #     return "Unable to find any recepient email address" 
    send_email(summary, recipient_name, recipient_email)   
    return


# MailGun Function to send Email
def send_email(summary: str, recipient_email: str, recipient_name: str=None) -> str:
    # Mailgun API endpoint
    url = f"https://api.mailgun.net/v3/oliver.solutions/messages"
    
    # Prepare the email data
    data = {
        "from": "admin@oliver.solutions",
        "to": recipient_email,
        "subject": "Document Summary",
        "text": f'Hi {recipient_name} \n\n {summary} \n\nRegards'
    }
    
    # Send the request to Mailgun
    response = requests.post(
        url,
        auth=(st.secrets['mailgun_user'], st.secrets['mailgun_api_key']),
        data=data
    )
    
    print(response)
    # Check the response status
    if response.status_code == 200:
        return "Email sent successfully! to " + recipient_email 
    else:
        return f"Failed to send email: {response.status_code}, {response.text}"


textSummarizerTool = FunctionTool.from_defaults(fn=textSummarizer)
emailSenderTool = FunctionTool.from_defaults(fn=emailSender)


# AI Agent context = """\
context = """\ You are an AI Agent who has assigned multiple tools one is for summarizing the text for sending it as a formal email and if tool is not working of summarizer provide the best content for email based on the user input by yourself\
   and other tool is for sending that summarize text as an email to user provided email address\
"""
# Initialize Agents
ai_agent = ReActAgent.from_tools(
    [textSummarizerTool, emailSenderTool],
    llm=llm,
    # context=context,
    verbose=True
)

# message = 'Can you please sumarize the following content in proper manner: 1- Fixed the colors scheme to a light tone, 2- Make the design more appealing 3. Make the colors vibrant so it will look attractive on banner'
# response = ai_agent.chat(message)
# print(str(response))

# message = 'Can you please send the above email to Steven at steve.rogers@avengers.com'
# response = ai_agent.chat(message)
# print(str(response))



# @cl.on_chat_start
# async def start():
#     cl.user_session.set('omgagent', ai_agent)
#     await cl.Message(content='Hello there!, I am OMG AI Agent. How can I help you?').send()
    
# @cl.on_message
# async def on_message(message: cl.Message):
#     omg_agent = cl.user_session.get('omgagent')
    
#     response = omg_agent.chat(message.content)

#     print(response)
#     print (message.content)
#     await cl.Message(content=str(message.content)).send()


# Streamed response emulator
def response_generator(response):
    # response = random.choice(
    #     [
    #         "Hello there! How can I assist you today?",
    #         "Hi! Is there anything I can help you with?",
    #         "Do you need help?",
    #     ]
    # )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title("OMG AI AGENT")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        agent_response = str(ai_agent.chat(prompt))
        # print((agent_response).split())
        response = st.write_stream(response_generator(agent_response))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})