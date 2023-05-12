import openai
import json
import re
from multiprocessing import Pool, freeze_support


GPT_4 = 'gpt-4'
GPT_3_5 = 'gpt-3.5-turbo'

class Dialogue():

    def __init__(self, char_therapist, char_visitor):
        self.char_therapist = char_therapist
        self.char_visitor = char_visitor
        self.history = []

    def call_openai(self, messages, model=GPT_3_5):
        response = openai.ChatCompletion.create(
            model=model,
            temperature=0.8,
            max_tokens=3600,
            messages=messages,
        )
        # print(f"message sent to openai ====== \n\n{messages}\n\n")
        return response


    def run_therapist(self):
        char_therapist = "Assume you are a professional Therapist, the user is a Visitor to you. You guide the visitor by asking questions based on their backgroud and experiences, instead of pvoviding lots of suggestions. The Visitor recently lost their job."
        messages = [{"role": "system", "content": char_therapist}]
        for i in range(len(self.history)):
            if i%2==0:
                messages.append({"role": "assistant", "content": self.history[i]})
            else:
                messages.append({"role": "user", "content": self.history[i]})

        openai_response = self.call_openai(messages)
        ai_message = openai_response["choices"][0]["message"]["content"]
        print(f"\n{ai_message}\n")
        self.history.append(ai_message)


    def run_visitor(self):
        char_visitor = "Assume you are a Visitor, and the user is a Therapist. You recently lost your job."
        messages = [{"role": "system", "content": char_visitor}]
        for i in range(len(self.history)):
            if i%2==1:
                messages.append({"role": "assistant", "content": self.history[i]})
            else:
                messages.append({"role": "user", "content": self.history[i]})

        openai_response = self.call_openai(messages)
        ai_message = openai_response["choices"][0]["message"]["content"]
        print(f"\n{ai_message}\n")
        self.history.append(ai_message)


    def generate_dialogue(self):
        self.history = []
        for i in range(4):
            self.run_therapist()
            self.run_visitor()
    

if __name__=="__main__":
    t = "Assume you are a professional Therapist, the user is a Visitor to you. The Visitor recently lost their job."
    v = "Assume you are a Visitor, and the user is a Therapist. You recently lost your job."
    d = Dialogue(char_therapist=t, char_visitor=v)
    d.generate_dialogue()