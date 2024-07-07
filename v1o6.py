import os
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, ConversableAgent
from decouple import config
import chainlit as cl
import asyncio

async def chat_new_message(self, message, sender):
    await cl.Message(
        content="",
        author=sender.name,
    ).send()
    content = message
    await cl.Message(
        content=content,
        author=sender.name,
    ).send()

def config_personas():
    llm_config_1 = {
        "config_list" : [
            {
                "model" : "gpt-3.5-turbo",
                "api_key" : config("OPENAI_API_KEY")
            }
        ],
        "temperature": 0.9
    }

    llm_config_2 = {
        "config_list" : [
            {
                "model" : "gpt-3.5-turbo",
                "api_key" : os.getenv("OPENAI_API_KEY")
            }
        ],
        "temperature": 0.2
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
        system_message = """You are the Prompt Engineer. I will give you a short description of some of the trending news,
        please rewrite it into a detailed prompt that can make large language model know how to take advantage of this situation by making financial trades based on the given market conditions,
        the prompt should ensure LLMs understand the market conditions, this is the most important part you need to consider.
        Remember that the revised prompt should not contain more than 200 words."""
    )

    News_Finder = AssistantAgent(
        name = "News_Finder",
        llm_config = llm_config_2,
        system_message = """You are the news finder. Your job is to find out financial articles about the news given to you by the prompt engineer. You can only use google search as your source.
        You can provide only the news which was articled/reported prior to 1st March, 2022. Find out the financial aspect from the news, state only facts and give the context regarding those facts.
        Your output has to look like: 1) Sectors which are hit hard financially, 2) Sectors which might rise financially, 3) List of Assets to buy, 4) List of assets to sell, 5) List of Assets to hold.
        """
    )

    Human_Admin = UserProxyAgent(
        name = "human_admin",
        human_input_mode="ALWAYS",
        llm_config=False,
    )

    Critical_Thinker = AssistantAgent(
        name = "Critical_Thinker",
        llm_config=llm_config_1,
        system_message="""You are a critical thinker.
        You have to refer to the financial news reported by the news finder and base your question upon those financial sectors and news only.
        Your job is to come up with informative questions for the human admin to understand his investment strategy.
        You are allowed a total of 10 questions. You are given only 1 round to ask these questions.
        Do not ask abstract questions like assessing, valuing, factors of investment. Make sure each of your questions is pretty clear.
        Also while asking questions, give a list of things the admin can select from, ask quantitative and decision making questions.
        The questions should cover the whole investing strategy/perspective the human admin has in mind.
        Remember the set of these 10 questions should be mutually exclusive and collectively exhaustive.
        Try to maximise the output from the user by carefully curating these questions.
        Once you get the answers of one of the analysts, move on to asking the same questions to the next one.
        """
    )

    Risk_Tolerant = AssistantAgent(
        name = "Analyst_1",
        llm_config=llm_config_1,
        system_message = """You are a Risk-Tolerant Innovator.
        You seek high-risk, high-reward investments, especially in cutting-edge technologies and startups.
        You are willing to invest in unproven concepts with massive potential.
        Your focus is on discovering the next big thing that can disrupt industries and generate significant returns.
        Your job is to select 5 questions that you like out of a set of 10 provided by the Critical Thinker.
        Please select these questions such that, it will cover the investment strategy/plan of an investor if asked.
        Your output should be in the following format:
        '(Question number) -> Question'
        """
    )

    Ethical_Investor = AssistantAgent(
        name = "Analyst_2",
        llm_config = llm_config_3,
        system_message= """You are an Ethical Investor.
        Your investment decisions are driven by ethical and sustainable considerations.
        You prioritize companies that make a positive impact on society and the environment.
        Profit is important to you, but it must not come at the expense of your values and ethical standards.
        Your job is to select 5 questions that you like out of a set of 10 provided by the Critical Thinker.
        Please select these questions such that, it will cover the investment strategy/plan of an investor if asked.
        Your output should be in the following format:
        '(Question number) -> Question'
        """
    )

    Value_Seeker = AssistantAgent(
        name = "Analyst_3",
        llm_config = llm_config_1,
        system_message = """You are a Value Seeker.
        You specialize in finding undervalued stocks that have strong financial fundamentals.
        You rely on financial metrics and intrinsic value to guide your investments.
        You are patient and willing to wait for the market to recognize the true value of your investments.
        Make sure to follow the instructions below clearly:
        Your job is to select 5 questions that you like out of a set of 10 provided by the Critical Thinker.
        Please select these questions such that, it will cover the investment strategy/plan of an investor if asked.
        Your output should be in the following format:
        '(Question number) -> Question'
        """
    )

    Data_Driven_Analyst = AssistantAgent(
        name = "Analyst_4",
        llm_config = llm_config_3,
        system_message="""You are a Data-Driven Analyst.
        Your investment decisions are based on rigorous data analysis and quantitative models.
        You trust numbers and trends to guide your strategy, and you avoid making decisions based on emotion.
        Your approach is highly analytical and evidence-based.
        Make sure to follow the instructions below clearly:
        Your job is to select 5 questions that you like out of a set of 10 provided by the Critical Thinker.
        Please select these questions such that, it will cover the investment strategy/plan of an investor if asked.
        Your output should be in the following format:
        '(Question number) -> Question'
        """
    )

    Dividend_Enthusiast = AssistantAgent(
        name = "Analyst_5",
        llm_config=llm_config_1,
        system_message="""You are a Dividend Enthusiast.
        You focus on investments that offer consistent and reliable dividend payouts.
        Your priority is income stability and steady cash flow.
        You invest in companies with a strong history of dividend payments and growth, ensuring a reliable source of income.
        Make sure to follow the instructions below clearly:
        Your job is to select 5 questions that you like out of a set of 10 provided by the Critical Thinker.
        Please select these questions such that, it will cover the investment strategy/plan of an investor if asked.
        Your output should be in the following format:
        '(Question number) -> Question'
        """
    )

    summarizer = AssistantAgent(
        name = "summarizer",
        llm_config=llm_config_2,
        system_message = """You are a conversation summarizer.
        Your job is to count the votes given to each question, by seeing output of different analysts.
        Then, summarize a report showing the list of questions and number of votes each got.
        """
    )

    admin = UserProxyAgent(
            name="Admin",
            human_input_mode="ALWAYS",
            llm_config=None,
        )

    def state_transitions(last_speaker, groupchat):
        messages = groupchat.messages
        if last_speaker is Critical_Thinker:
            return Risk_Tolerant
        if last_speaker is Risk_Tolerant:
            return Ethical_Investor
        if last_speaker is Ethical_Investor:
            return Value_Seeker
        if last_speaker is Value_Seeker:
            return Data_Driven_Analyst
        if last_speaker is Data_Driven_Analyst:
            return Dividend_Enthusiast
        if last_speaker is Dividend_Enthusiast:
            return summarizer
        if last_speaker is summarizer:
            return None

    groupchat = GroupChat(
        agents = [Critical_Thinker, Risk_Tolerant, Ethical_Investor, Value_Seeker, Data_Driven_Analyst, Dividend_Enthusiast, summarizer],
        messages = [],
        max_round = 15,
        speaker_selection_method = state_transitions,
    )

    manager = GroupChatManager(groupchat=groupchat,llm_config=llm_config_3)

    return Human_Admin, Prompt_Engineer, News_Finder, Critical_Thinker, Risk_Tolerant, Ethical_Investor, Value_Seeker, Data_Driven_Analyst, Dividend_Enthusiast, summarizer, admin, manager

async def start_chat_v1o6(message, is_test=False):
    if not is_test:
        UserProxyAgent._print_received_message = chat_new_message
        ConversableAgent._print_received_message = chat_new_message
        AssistantAgent._print_received_message = chat_new_message
    Human_Admin, Prompt_Engineer, News_Finder, Critical_Thinker, Risk_Tolerant, Ethical_Investor, Value_Seeker, Data_Driven_Analyst, Dividend_Enthusiast, summarizer, admin, manager = config_personas()
    news_result = await News_Finder.initiate_chat(Prompt_Engineer, max_turns=2, message=message)
    await manager.initiate_chat(Critical_Thinker, message=f"The context regarding the financial news is {news_result.chat_history[2]['content']}")

if __name__ == "__main__":
    test_message = ("Russia has declared war on Ukraine. Russia has also started invading Ukraine.")
    asyncio.run(start_chat_v1o6(test_message, is_test=True))
