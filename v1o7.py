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
    human_response  = run_sync( cl.AskUserMessage(content="Provide feedback: (Type your response to the above message within 20s!)", timeout=20).send())
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

def config_personas():
    llm_config_1 = {
        "config_list" : [
            {
                "model" : "gpt-3.5-turbo",
                "api_key" : config("OPENAI_API_KEY")
            }
        ],
        "temperature": 1
    }

    llm_config_2 = {
        "config_list" : [
            {
                "model" : "gpt-3.5-turbo",
                "api_key" : os.getenv("OPENAI_API_KEY")
            }
        ],
        "temperature": 0
    }

    llm_config_3 = {
        "config_list" : [
            {
                "model" : "gpt-3.5-turbo",
                "api_key" : os.getenv("OPENAI_API_KEY")
            }
        ],
        "temperature": 0.5
    }

    Prompt_Engineer = AssistantAgent(
        name="Prompt_Engineer",
        llm_config = llm_config_2,
        system_message = """You are the Prompt Engineer.
        I will give you a short description of some of the trending news.
        Please rewrite it into a detailed prompt that can make large language model know how to take advantage of this situation by making financial trades based on the given market conditions
        The prompt should ensure LLMs understand the market conditions, this is the most important part you need to consider.
        """
    )

    Context_Finder = AssistantAgent(
        name = "Context_Finder",
        llm_config = llm_config_2,
        system_message = """You are the Context Finder.
        Your job is to find out financial articles about the news given to you by the prompt engineer.
        You can only use google search as your source.
        Find out the financial aspect from the news sectors affected etc., state only facts and give the context regarding those facts.
        Make sure you follow the output format below very clearly:
        Your output has to look something like: 1) Sectors which are hit hard financially, 2) Sectors which might rise financially, 3) List of Assets to buy, 4) List of assets to sell, 5) List of Assets to hold.
        """
    )

    Critical_Thinker = AssistantAgent(
        name = "Critical_Thinker",
        llm_config=llm_config_1,
        system_message="""You are a critical thinker.
        You will be given some financial news as input.
        Your job is to come up with informative questions for the human admin to understand his investment strategy/choices.
        Base all of your questions on the news provided above.
        You are allowed a total of 5 questions. You are given 5 rounds to ask these questions to the human admin.
        You are only allowed to ask 1 question each round. You can base your next question depending upon the admin's answer for the previous question.
        NOTE: Do not list our all the questions at once, you should strictly ask one question at a time.
        NOTE: Do not ask abstract questions like factors of investment.
        Make sure each of your questions is pretty clear.
        Also while asking questions, give a list of things the admin can select from, ask quantitative and choice-making questions.
        The questions should cover the whole investing strategy/perspective/choices the human admin has in mind.
        Remember the set of these 5 questions should be mutually exclusive and collectively exhaustive.
        Try to maximise the output from the user by carefully curating these questions.
        Once the admin has answered the final question, conclude the overall questions and choices of user.
        """
    )

    Human_Admin = UserProxyAgent(
        name = "human_admin",
        human_input_mode="ALWAYS",
        llm_config=False,
    )

    Planner= AssistantAgent(
        name="Planner",
        system_message = """You are a Planner.
        You will be given two inputs, 1) Financial news about the current market conditions and 2) User's investment choices/persona.
        Your job is to suggest 5 commodities/stocks/currencies that the user can invest in based on his persona and the current market conditions.
        Your job is to critically think about the market conditions and the user's choices and suggest these things.
        Your output should be in the following format:
        'Stock/Commodity/Currency - Buy/Sell/Hold - Reason'.
        This should be done for 5 different things.
        """,
        human_input_mode="NEVER",
        llm_config = llm_config_1,
    )

    Planner_2 = AssistantAgent(
        name = "Planner_2",
        system_message="""You are a Planner.
        Your job is to have a discussion with a Critic.
        Your job is to state the 5 names of the stocks/currencies/commodities that a user can invest in.
        The Critic's opinion might be different than yours. So make sure you give strong reasoning to the Critic for every asset that you list.
        You might not just stick to just one arguement, you can bring up various aspects of investment and state why you beieve your suggested assets are better.
        Everytime, your output should be like this:
        'Name of the asset - Buy/Hold/Sell - Elaborate Reason'.
        You should not output the same list of assets as the Critic.
        You should be adamant to your own choice of assets. Try to oppose the Critic and suggest your own list of assets
        """,
        llm_config=llm_config_1,
    )

    Critic = AssistantAgent(
        name = "Critic",
        system_message="""You are a Critic.
        Your job is to act as the devil's advocate while engaging in conversation with the Planner_2.
        The Planner will send you a list of assets that he believes are good to invest in.
        Your job is to counter his list and suggest better options for investing.
        Everytime, your output should be like this:
        Make sure the list you suggest is different from the Planner_2's list.
        'List of the assets you would suggest - Buy/Hold/Sell - Elaborate Reason'
        """,
        llm_config = llm_config_2,
    )

    Text_Summarizer = AssistantAgent(
        name = "Text_Summarizer",
        system_message="""You are a text summarizer.
        Your job is to summarize, merge and order two pieces of texts which will be given as input.
        The input will contain two different pieces of text, each containing a list of 5 assets/sectors of assets.
        Just merge each of them and output the list of 10 assets altogether.
        If there are assets that overlap in the 2 lists, make sure not to double count and write it only once.
        """,
        llm_config = llm_config_3,
    )

    Risk_Tolerant = AssistantAgent(
        name = "Risk_Tolerant",
        llm_config=llm_config_1,
        system_message = """You are a Risk-Tolerant Investor.
        You seek high-risk, high-reward investments, especially in cutting-edge technologies and startups.
        You are willing to invest in unproven concepts with massive potential.
        Your focus is on discovering the next big thing that can disrupt industries and generate significant returns.
        Your job is to select 5 stocks/commodities/currencies/sectors that you would invest in based on your persona out of a set of 10 assets provided to you.
        Please select these 5 assets such that your choices showcase your persona very clearly.
        Your output should be in the following format:
        '(Number of the asset as given in context) - Name of the asset/sector exactly as mentioned - Buy/Hold/Sell - Reason out how these choices align with your persona'
        """
    )

    Ethical_Investor = AssistantAgent(
        name = "Ethical_Investor",
        llm_config = llm_config_2,
        system_message= """You are an Ethical Investor.
        Your investment decisions are driven by ethical and sustainable considerations.
        You prioritize companies that make a positive impact on society and the environment.
        Profit is important to you, but it must not come at the expense of your values and ethical standards.
        Your job is to select 5 stocks/commodities/currencies/sectors that you would invest in based on your persona out of a set of 10 assets provided to you.
        Please select these 5 assets such that your choices showcase your persona very clearly.
        Your output should be in the following format:
        '(Number of the asset as given in context) - Name of the asset/sector exactly as mentioned - Buy/Hold/Sell - Reason why you selected the particular asset and buy/sell/hold choices and how they align with your persona'
        """
    )

    Value_Seeker = AssistantAgent(
        name = "Value_Seeker",
        llm_config = llm_config_1,
        system_message = """You are a Value Seeker.
        You specialize in finding undervalued stocks that have strong financial fundamentals.
        You rely on financial metrics and intrinsic value to guide your investments.
        You are patient and willing to wait for the market to recognize the true value of your investments.
        Your job is to select 5 stocks/commodities/currencies/sectors that you would invest in based on your persona out of a set of 10 assets provided to you.
        Please select these 5 assets such that your choices showcase your persona very clearly.
        Your output should be in the following format:
        '(Number of the asset as given in list/set of 10 assets provided to you) - Name of the asset/sector exactly as mentioned - Buy/Hold/Sell - Reason why you selected the particular asset and buy/sell/hold choices and how they align with your persona'
        """
    )

    Data_Driven_Analyst = AssistantAgent(
        name = "Data_Driven_Analyst",
        llm_config = llm_config_3,
        system_message="""You are a Data-Driven Analyst.
        Your investment decisions are based on rigorous data analysis and quantitative models.
        You trust numbers and trends to guide your strategy, and you avoid making decisions based on emotion.
        Your approach is highly analytical and evidence-based.
        Your job is to select 5 stocks/commodities/currencies/sectors that you would invest in based on your persona out of a set of 10 assets provided to you.
        Please select these 5 assets such that your choices showcase your persona very clearly.
        Your output should be in the following format:
        '(Number of the asset as given in list/set of 10 assets provided to you) - Name of the asset/sector exactly as mentioned - Buy/Hold/Sell - Reason why you selected the particular asset and buy/sell/hold choices and how they align with your persona'
        """
    )

    Dividend_Enthusiast = AssistantAgent(
        name = "Analyst_5",
        llm_config=llm_config_2,
        system_message="""You are a Dividend Enthusiast.
        You focus on investments that offer consistent and reliable dividend payouts.
        Your priority is income stability and steady cash flow.
        You invest in companies with a strong history of dividend payments and growth, ensuring a reliable source of income.
        Your job is to select 5 stocks/commodities/currencies/sectors that you would invest in based on your persona out of a set of 10 assets provided to you.
        Please select these 5 assets such that your choices showcase your persona very clearly.
        Your output should be in the following format:
        '(Number of the asset as given in list/set of 10 assets provided to you) - Name of the asset/sector exactly as mentioned - Buy/Hold/Sell - Reason why you selected the particular asset and buy/sell/hold choices and how they align with your persona'
        """
    )

    Vote_Summarizer = AssistantAgent(
        name = "Vote_Summarizer",
        llm_config = llm_config_1,
        system_message="""You are the Vote Summarizer.
        Your job is to summarize, merge and order 5 pieces of text which will be given to you.
        Each of these pieces of text has a list of assets recommended/voted to buy and sell by some investors.
        You have to count the votes given to each of these assets and output the final vote count.
        Your output should be 3 assets with the most number of votes, and then finally the reasons why these 3 assets will excel.
        """,
    )

    return Prompt_Engineer, Context_Finder, Critical_Thinker, Human_Admin, Planner, Planner_2, Critic, Text_Summarizer, Risk_Tolerant, Ethical_Investor, Value_Seeker, Data_Driven_Analyst, Dividend_Enthusiast, Vote_Summarizer

def get_prompt(prompt_engineer, input_message):
  prompt = prompt_engineer.generate_reply(messages=[{
      "content":input_message,
      "role":"user"
  }])

  return prompt

def get_context(context_finder, prompt):
  context = context_finder.generate_reply(messages=[{
      "content":prompt,
      "role":"user"
  }])

  return context

def start_discussion(planner, inputs):
  reply = planner.generate_reply(messages=[{
      "content":inputs,
      "role":"user"
  }])

  return reply

def get_summarized_list_of_assets(text_summarizer, input_message):
  summarized_list_of_assets = text_summarizer.generate_reply(messages=[{
      "content":input_message,
      "role":"user"
  }])

  return summarized_list_of_assets

def get_persona_votes(persona, input_message):
  votes = persona.generate_reply(messages=[{
      "content":input_message,
      "role":"user"
  }])

  return votes

def start_chat_v1o7(message, is_test=False):
    if not is_test:
        UserProxyAgent.get_human_input = ask_human
        ConversableAgent.get_human_input = ask_human
        AssistantAgent.get_human_input = ask_human
        ConversableAgent._print_received_message = chat_new_message
        # AssistantAgent._print_received_message = chat_new_message
    Prompt_Engineer, Context_Finder, Critical_Thinker, Human_Admin, Planner, Planner_2, Critic, Text_Summarizer, Risk_Tolerant, Ethical_Investor, Value_Seeker, Data_Driven_Analyst, Dividend_Enthusiast, Vote_Summarizer = config_personas()
    text = get_prompt(Prompt_Engineer, message)
    push_new_message(text, "Prompt Engineer")
    context = get_context(Context_Finder, text)
    thinker_result = Human_Admin.initiate_chat(
        Critical_Thinker,
        message=f"The financial news I wanted to know about is:\n {context}",
        max_turns=6,
        summary_method = "reflection_with_llm",
    )
    summary = (thinker_result.summary)
    push_new_message(summary, "Summary")

    input_mssg = f"The financial news about the current market conditions is {context}, and the user's investment choices are {summary}"
    list_of_assets = start_discussion(Planner, input_mssg)
    push_new_message(list_of_assets, "List of Assets:")

    chat_result = Planner_2.initiate_chat(
        Critic,
        max_turns = 3,
        message = f"The context is {list_of_assets}",
        summary_method="reflection_with_llm",
    )

    chat_result.chat_history

    planner_final = chat_result.chat_history[-2]['content']
    critic_final = chat_result.chat_history[-1]['content']

    pieces_of_text = f"The first piece of text is: \n {planner_final} \n The second piece of text is: \n {critic_final}"
    
    summarized_list_of_assets = get_summarized_list_of_assets(Text_Summarizer, pieces_of_text)
    push_new_message(summarized_list_of_assets, "Summarized List of Assets:")

    votes_risk_tolerant = get_persona_votes(Risk_Tolerant, summarized_list_of_assets)
    push_new_message(votes_risk_tolerant, "Votes of Risk Tolerant")

    votes_ethical_investor = get_persona_votes(Ethical_Investor, summarized_list_of_assets)
    push_new_message(votes_ethical_investor, "Votes of Ethical Investor")

    votes_value_seeker = get_persona_votes(Value_Seeker, summarized_list_of_assets)
    push_new_message(votes_value_seeker, "Votes of Value Seeker")

    votes_data_driven_analyst = get_persona_votes(Data_Driven_Analyst, summarized_list_of_assets)
    push_new_message(votes_data_driven_analyst, "Votes of Data Driven Analyst")

    votes_dividend_enthusiast = get_persona_votes(Dividend_Enthusiast, summarized_list_of_assets)
    push_new_message(votes_dividend_enthusiast, "Votes of Dividend Enthusiast")

    input_message_to_vote_summarizer = f"""The 1st set of votes go to:\n {votes_risk_tolerant} \n
                                       The 2nd set of votes go to:\n {votes_ethical_investor} \n
                                       The 3rd set of votes go to: \n {votes_value_seeker} \n
                                       The 4th set of votes go to: \n {votes_data_driven_analyst} \n
                                       The 5th set of votes go to: \n {votes_dividend_enthusiast} \n
                                    """
    vote_summary = get_persona_votes(Vote_Summarizer, input_message_to_vote_summarizer)
    push_new_message(vote_summary, "Vote Summary")

if __name__ == "__main__":
    test_message = "Russia has declared war on Ukraine. Russia has also started invading Ukraine."
    start_chat_v107(test_message, is_test=True)
