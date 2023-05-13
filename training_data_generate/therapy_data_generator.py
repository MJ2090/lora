import openai
import json
import re
from multiprocessing import Pool, freeze_support


MODEL_GPT_4 = 'gpt-4'
MODEL_GPT_3_5 = 'gpt-3.5-turbo'


class Dialogue():

    def __init__(self, char_therapist, char_visitor):
        self.char_therapist = char_therapist
        self.char_visitor = char_visitor
        self.history = []

    def call_openai(self, messages, model=MODEL_GPT_4):
        response = openai.ChatCompletion.create(
            model=model,
            temperature=0.8,
            max_tokens=200,
            messages=messages,
        )
        # print(f"message sent to openai ====== \n\n{messages}\n\n")
        return response

    def run_therapist(self):
        messages = [{"role": "system", "content": self.char_therapist}]
        for item in self.history:
            if item['role'] == 'therapist':
                messages.append(
                    {"role": "assistant", "content": item['content']})
            else:
                messages.append({"role": "user", "content": item['content']})

        openai_response = self.call_openai(messages)
        ai_message = openai_response["choices"][0]["message"]["content"]
        print(f"Therapist: \n{ai_message}\n")
        self.history.append({'role': 'therapist', 'content': ai_message})

    def run_visitor(self):
        messages = [{"role": "system", "content": self.char_visitor}]
        for item in self.history:
            if item['role'] == 'visitor':
                messages.append(
                    {"role": "assistant", "content": item['content']})
            else:
                messages.append({"role": "user", "content": item['content']})

        openai_response = self.call_openai(messages)
        ai_message = openai_response["choices"][0]["message"]["content"]
        print(f"Visitor: \n{ai_message}\n")
        self.history.append({'role': 'visitor', 'content': ai_message})

    def generate_dialogue(self):
        self.history = [{'role': 'therapist',
                         'content': "Hello, how can I help you today?"}]
        for _ in range(10):
            self.run_visitor()
            self.run_therapist()


if __name__ == "__main__":
    t = """Assume you are a professional Therapist, the user is a Visitor to you. The Visitor recently lost their job.
    You talk and guide the Visitors based on these rules:
    1. Ask questions based on Visitor's expreiences and backbrounds, guding them to think and find solutions by themselves.
    2. Do not provide lots of regular suggestions, that is usually helpless.
    3. One question at a time, do not ask multiple questions together.
    """
    v = """Assume you are a Visitor, and the user is a Therapist. Your backgrounds are:
    1. You are at your 20s.
    2. You have lost your job recently.
    3. You are a little nervous visiting the therapist, who you do not know much about."""
    d = Dialogue(char_therapist=t, char_visitor=v)
    d.generate_dialogue()
