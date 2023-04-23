import json

file_names = ['question-2023-04-22-0.question',
             'question-2023-04-22-1.question', 
             'question-2023-04-23-2.question', 
             'question-2023-04-23-3.question',
             'question-2023-04-23-4.question', 
             'question-2023-04-23-5.question', 
             'question-2023-04-23-6.question', 
             'question-2023-04-23-7.question']

output_file = 'question.json'

def clean():
    all_json = []
    for f_name in file_names:
        f = open(f'training_data/{f_name}', 'r')
        s = json.loads(f.read())
        for item in s:
            print(item)
            all_json.append({'instruction': 'answer the question in the input.', 'input': item['question:'], 'output': item['answer']})

    f = open(f'training_data/{output_file}', 'w')
    f.write(json.dumps(all_json))


if __name__ == "__main__":
    clean()

