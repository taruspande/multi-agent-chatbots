import os
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, ConversableAgent
from decouple import config
import chainlit as cl
from chainlit import run_sync
from autogen.graph_utils import visualize_speaker_transitions_dict
from autogen.coding import LocalCommandLineCodeExecutor
import random
import matplotlib.pyplot as plt
import networkx as nx
import json
import asyncio
import nest_asyncio

nest_asyncio.apply()

llm_config_1 = {
    "config_list": [
        {
            "model": "gpt-3.5-turbo",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    ],
    "temperature": 1
}

llm_config_2 = {
    "config_list": [
        {
            "model": "gpt-3.5-turbo",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    ],
    "temperature": 0
}

llm_config_3 = {
    "config_list": [
        {
            "model": "gpt-3.5-turbo",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    ],
    "temperature": 0.5
}


def chat_new_message(self, message, sender):
    if sender.name == "human_admin":
        return
    cl.run_sync(
        cl.Message(
            content="",
            author=sender.name,
        ).send()
    )
    content = message
    cl.run_sync(
        cl.Message(
            content=content,
            author=sender.name,
        ).send()
    )

def ask_human(self, prompt: str) -> str:
    human_response = run_sync(cl.AskUserMessage(content="Provide feedback: (Type your response to the above message within 20s!)", timeout=20).send())
    if human_response:
        return human_response["output"]
    else:
        return "No response"

def push_new_message(message, sender):
    cl.run_sync(
        cl.Message(
            content=message,
            author=sender,
        ).send()
    )

def get_llama(temperature=0.7):
    llm_config_llama = {
        "config_list": [
            {
                "model": "meta-llama/Llama-3-8b-chat-hf",
                "api_key": os.getenv("TOGETHER_API_KEY"),
                "base_url": "https://api.together.xyz/v1"
            }
        ],
        "temperature": temperature,
    }
    return llm_config_llama

def get_dbrx(temperature=0.7):
    llm_config_dbrx = {
        "config_list": [
            {
                "model": "databricks/dbrx-instruct",
                "api_key": os.getenv("TOGETHER_API_KEY"),
                "base_url": "https://api.together.xyz/v1"
            }
        ],
        "temperature": temperature,
    }
    return llm_config_dbrx

def get_mistral(temperature=0.7):
    llm_config_mistral = {
        "config_list": [
            {
                "model": "mistralai/Mistral-7B-Instruct-v0.1",
                "api_key": os.getenv("TOGETHER_API_KEY"),
                "base_url": "https://api.together.xyz/v1"
            }
        ],
        "temperature": temperature,
    }
    return llm_config_mistral

def load_config(file_path):
    """Load configuration from a JSON file."""
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def get_items_from_config(config_dict):
    output_dict = {}
    for key in config_dict.keys():
        if key == "num_agents":
            output_dict.update({key: config_dict[key]})
            continue

        model = config_dict[key][0]
        temp = float(config_dict[key][1])

        if model == 'llama':
            llm_config_dict = get_llama(temp)
            output_dict.update({key: llm_config_dict})

        elif model == 'dbrx':
            llm_config_dict = get_dbrx(temp)
            output_dict.update({key: llm_config_dict})

        elif model == 'mistral':
            llm_config_dict = get_mistral(temp)
            output_dict.update({key: llm_config_dict})

        else:
            continue

    return output_dict

config = load_config('config.json')
num_layers = config['num_layers']

ls_dicts = []
for i in range(num_layers):
    ls_dicts.append(config[f"Layer_{i+1}"])

def debate_flow(context, news, llm_config_dict):
    num_agents = llm_config_dict["num_agents"]
    debate_list = []
    for i in range(num_agents):
        agent = AssistantAgent(
            name=f"Agent_{i+1}",
            llm_config=llm_config_dict[f"Agent_{i+1}"],
            system_message="""You are a debator.
            You are not allowed to pause/interrupt the discussion.
            You should always give your opinions in the debate irrespective of the topic.
            You should give your opinion at least once.
            Never flip roles."""
        )
        debate_list.append(agent)

    groupchat = GroupChat(
        agents=debate_list,
        messages=[],
        max_round=10,
        role_for_select_speaker_messages="user",
    )

    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config_3)
    grpchat = manager.initiate_chat(
        debate_list[0], message=f"Debate if the following {context} are good for investment given the news {news}",
        summary_method="reflection_with_llm"
    )

    return grpchat

def risk_tolerant(index, llm_config_dict):
    agent = AssistantAgent(
        name=f"Agent_{index+1}",
        system_message="""You are a Risk-Tolerant Investor.
        You seek high-risk, high-reward investments, especially in cutting-edge technologies and startups.
        You are willing to invest in unproven concepts with massive potential.
        Your focus is on discovering the next big thing that can disrupt industries and generate significant returns.
        Your job is to recommend 5 stocks/commodities/currencies/sectors that you would invest in based on your persona, given a certain context.
        Please select these 5 assets such that your choices showcase your persona very clearly.""",
        llm_config=llm_config_dict[f"Agent_{index+1}"]
    )
    return agent

def ethical_investor(index, llm_config_dict):
    agent = AssistantAgent(
        name=f"Agent_{index+1}",
        system_message="""You are an Ethical Investor.
        Your investment decisions are driven by ethical and sustainable considerations.
        You prioritize companies that make a positive impact on society and the environment.
        Profit is important to you, but it must not come at the expense of your values and ethical standards.
        Your job is to recommend 5 stocks/commodities/currencies/sectors that you would invest in based on your persona, given a certain context.
        Please select these 5 assets such that your choices showcase your persona very clearly.""",
        llm_config=llm_config_dict[f"Agent_{index+1}"]
    )
    return agent

def value_seeker(index, llm_config_dict):
    agent = AssistantAgent(
        name=f"Agent_{index+1}",
        system_message="""You are a Value Seeker.
        You specialize in finding undervalued stocks that have strong financial fundamentals.
        You rely on financial metrics and intrinsic value to guide your investments.
        You are patient and willing to wait for the market to recognize the true value of your investments.
        Your job is to recommend 5 stocks/commodities/currencies/sectors that you would invest in based on your persona, given a certain context.
        Please select these 5 assets such that your choices showcase your persona very clearly.""",
        llm_config=llm_config_dict[f"Agent_{index+1}"]
    )
    return agent

def data_driven_analyst(index, llm_config_dict):
    agent = AssistantAgent(
        name=f"Agent_{index+1}",
        system_message="""You are a Data-Driven Analyst.
        Your investment decisions are based on rigorous data analysis and quantitative models.
        You trust numbers and trends to guide your strategy, and you avoid making decisions based on emotion.
        Your approach is highly analytical and evidence-based.
        Your job is to recommend 5 stocks/commodities/currencies/sectors that you would invest in based on your persona, given a certain context.
        Please select these 5 assets such that your choices showcase your persona very clearly.""",
        llm_config=llm_config_dict[f"Agent_{index+1}"]
    )
    return agent

def dividend_enthusiast(index, llm_config_dict):
    agent = AssistantAgent(
        name=f"Agent_{index+1}",
        system_message="""You are a Dividend Enthusiast.
        You focus on investments that offer consistent and reliable dividend payouts.
        Your priority is income stability and steady cash flow.
        You invest in companies with a strong history of dividend payments and growth, ensuring a reliable source of income.
        Your job is to recommend 5 stocks/commodities/currencies/sectors that you would invest in based on your persona, given a certain context.
        Please select these 5 assets such that your choices showcase your persona very clearly.""",
        llm_config=llm_config_dict[f"Agent_{index+1}"]
    )
    return agent

def planning_flow(news, llm_config_dict, context, summary):
    num_agents = llm_config_dict["num_agents"]
    Vote_Summarizer = AssistantAgent(
        name="Vote_Summarizer",
        system_message=f"""You are the Vote Summarizer.
        Your job is to summarize, merge and order {num_agents} pieces of text which will be given to you.
        Each of these pieces of text has a list of assets recommended/voted to buy and sell by some investors.
        Your output should have a list of all the assets the {num_agents} have provided.""",
        llm_config=llm_config_2
    )

    planners = []
    for i in range(num_agents):
        if i % 5 == 0:
            planners.append(risk_tolerant(i, llm_config_dict))
        elif i % 5 == 1:
            planners.append(ethical_investor(i, llm_config_dict))
        elif i % 5 == 2:
            planners.append(value_seeker(i, llm_config_dict))
        elif i % 5 == 3:
            planners.append(data_driven_analyst(i, llm_config_dict))
        else:
            planners.append(dividend_enthusiast(i, llm_config_dict))

    planners.append(Vote_Summarizer)

    groupchat = GroupChat(
        agents=planners,
        messages=[],
        max_round=10,
        role_for_select_speaker_messages="user",
    )

    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config_3)
    grpchat = manager.initiate_chat(
        planners[0], message=f"The financial news is \n{news}\n, the context provided is \n{context}. The user preferences are {summary}",
        summary_method="reflection_with_llm"
    )

    return grpchat

def config_personas():
    Prompt_Engineer = AssistantAgent(
        name="Prompt_Engineer",
        system_message="""You are the Prompt Engineer.
        I will give you a short description of some of the trending news.
        Please rewrite it into a detailed prompt that can make large language model know how to take advantage of this situation by making financial trades based on the given market conditions
        The prompt should ensure LLMs understand the market conditions, this is the most important part you need to consider.""",
        llm_config=get_mistral(0.8),
    )

    Context_Finder = AssistantAgent(
        name="Context_Finder",
        system_message="""You are the Context Finder.
        Your job is to find out financial articles about the news given to you by the prompt engineer.
        You can only use google search as your source.
        Find out the financial aspect from the news sectors affected etc., state only facts and give the context regarding those facts.
        Make sure you follow the output format below very clearly:
        Your output has to look something like: 1) Sectors which are hit hard financially, 2) Sectors which might rise financially, 3) List of Assets to buy, 4) List of assets to sell, 5) List of Assets to hold.""",
        llm_config=get_llama(1),
    )

    Critical_Thinker = AssistantAgent(
        name="Critical_Thinker",
        system_message="""You are a critical thinker.
        You will be given some financial news as input.
        Your job is to come up with informative questions for the human admin to understand his investment strategy/choices.
        Base all of your questions on the news provided above.
        You are allowed a total of 5 questions. You are given 5 rounds to ask these questions to the human admin.
        You are only allowed to ask 1 question each round. You can base your next question depending upon the admin's answer for the previous question.
        NOTE: Do not ask abstract questions like factors of investment.
        Make sure each of your questions is pretty clear.
        Also while asking questions, give a list of things the admin can select from, ask quantitative and choice-making questions.
        The questions should cover the whole investing strategy/perspective/choices the human admin has in mind.
        Remember the set of these 5 questions should be mutually exclusive and collectively exhaustive.
        Try to maximize the output from the user by carefully curating these questions.
        Once the admin has answered the final question, write the word 'TERMINATE'.""",
        llm_config=llm_config_1,
    )

    Human_Admin = UserProxyAgent(
        name="human_admin",
        human_input_mode="ALWAYS",
        llm_config=False,
    )

    return Prompt_Engineer, Context_Finder, Critical_Thinker, Human_Admin

def get_prompt(prompt_engineer, input_message):
    prompt = prompt_engineer.generate_reply(messages=[{
        "content": input_message,
        "role": "user"
    }])
    return prompt

def get_context(context_finder, prompt):
    context = context_finder.generate_reply(messages=[{
        "content": prompt,
        "role": "user"
    }])
    return context

def get_list_of_assets(grpchat):
    i = -1
    while True:
        if (grpchat.chat_history[i])["name"] == "Vote_Summarizer":
            output = grpchat.chat_history[i]["content"]
            break
        i -= 1
    return output

def start_chat_v1o8(message, is_test=False):
    if not is_test:
        UserProxyAgent.get_human_input = ask_human
        ConversableAgent.get_human_input = ask_human
        AssistantAgent.get_human_input = ask_human
        ConversableAgent._print_received_message = chat_new_message

    Prompt_Engineer, Context_Finder, Critical_Thinker, Human_Admin = config_personas()

    news = get_prompt(Prompt_Engineer, message)
    push_new_message(news, "Prompt_Engineer")

    context = get_context(Context_Finder, news)
    push_new_message(context, "Context_Finder")

    thinker_result = Human_Admin.initiate_chat(
        Critical_Thinker,
        message=f"The financial news I wanted to know about is:\n {context}",
        max_turns=10,
        summary_method="reflection_with_llm",
    )
    summary = thinker_result.summary
    push_new_message(summary, "Critical_Thinker")

    num_layers = config['num_layers']
    ls_layers = [config[f"Layer_{i+1}"] for i in range(num_layers)]

    for i in range(num_layers):
        llm_config_dictionary = get_items_from_config(ls_layers[i])
        if i % 2 == 0:
            grpchat_1 = planning_flow(news, llm_config_dictionary, context, summary)
            context = get_list_of_assets(grpchat_1)
            if i == num_layers - 1:
                push_new_message(context, "Planning Flow")
        else:
            grpchat_2 = debate_flow(context, news, llm_config_dictionary)
            summary = grpchat_2.summary
            if i == num_layers - 1:
                push_new_message(summary, "Debate Flow")

if __name__ == "__main__":
    test_message = "Russia has declared war on Ukraine. Russia has also started invading Ukraine."
    start_chat_v1o8(test_message, is_test=True)
