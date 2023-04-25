import openai
import json
import re
from multiprocessing import Pool, freeze_support


patients = ['lost his job recently, which was not due to his fault.',
            'lost his parents recently,  which loved him so much.',
            'lost his only son recently.',
            'lost his best friend he has known for years recently.',
            'lost his wife recently.',
            'lost his husband recently.',
            'suffers from work pressure from his managers.',
            'suffers from a huge lose in the stock market.',
            'suffers from depression.',
            'suffers from anxiety.',
            'suffers from a divorce which was not her fault.',
            'broke up with her boyfriend.',
            'broke up with his girlfriend.',
            'failed to pass the final exam in college.',
            'has trouble sleeping and has lost weight.',
            'recently went through a divorce and is struggling to adjust to his new life.',
            'has been experiencing high levels of stress at work, which is impacting his mental health and well-being.',
            'feels constantly worried and anxious, and finds it hard to relax or enjoy herself.',
            'feels sad and lonely, and finds it hard to engage in activities he used to enjoy.',
            'has been diagnosed with bipolar disorder and is struggling to manage her symptoms.',
            'has experienced manic episodes and is concerned about the impact on her relationships and work.',
            'has been struggling with addiction to alcohol and is ready to seek help.',
            'has been diagnosed with a chronic illness and is struggling to cope with the physical and emotional impact.',
            'has been experiencing conflict in her marriage and is concerned about the impact on her relationship.',
            'has experienced trauma in his past and is struggling to cope with the emotional impact.',
            'has been struggling with low self-esteem and body image issues.',
            'has been experiencing discrimination and harassment in his workplace due to his sexual orientation.',
            'had a baby and is struggling to adjust to motherhood.',
            'has been struggling with feelings of guilt and shame related to a past mistake where he had a car accident.',
            'recently went through a breakup and is struggling to cope with her emotions.',
            'has been experiencing intense anger and impulsivity, which is impacting his relationships and work.',
            'has been feeling "stuck" in her life and is struggling to find direction and purpose.',
            'has been struggling with a phobia of flying, which is impacting his ability to travel and enjoy life.',
            'has been struggling with disordered eating and is concerned about the impact on her health.',
            'has just moved to another city and feels uncomfortable.',
            'has just started their college and feels uncomfortable.',
            'often quarrels with her parents a lot.',
            'often quarrels with her managers a lot.',
            'often quarrels with her two sons a lot.',
            'is facing discrimination or societal oppression related to their identity',
            'just became a father and feels uncomfortable',
            'just became a mother and feels uncomfortable',
            'is experiencing difficulty managing anger or impulsivity.',
            'has been struggling with low mood and lack of motivation for several months.',
            'has been experiencing panic attacks and feels like he is losing control of his life.',
            'has been diagnosed with an eating disorder and is struggling to manage her symptoms.',
            'has been struggling with addiction to drugs and alcohol and is ready to seek help.',
            'has been experiencing intrusive thoughts and feels like she is losing her mind.',
            'has been struggling with perfectionism and feels like she is never good enough.',
            'has been experiencing symptoms of OCD and is having trouble managing his compulsions.',
            'has been struggling with social anxiety and feels like it is impacting her ability to live a full life.',
            'has been experiencing symptoms of depression and feels like he has lost his sense of purpose in life.',
            'has been struggling with grief and loss after the death of a loved one.',
            'has been struggling with chronic pain and is having trouble managing the emotional impact.',
            'has been experiencing symptoms of ADHD and is having trouble managing his time and staying focused.',
            'has been struggling with infertility and is feeling overwhelmed by her emotions.',
            'has been experiencing symptoms of PTSD after a traumatic event and is struggling to cope with the emotional impact.',
            'has been experiencing symptoms of bipolar disorder and is struggling to manage her moods.',
            'has been struggling with relationship issues and feels like he has stuck in a pattern of toxic relationships.',
            'has been experiencing symptoms of insomnia and is having trouble sleeping.',
            'suffers from PTSD after serving for 3 years in the army in Afghanistan.',
            'suffers from PTSD after being involved in a terrible car accident.',
            'suffers from PTSD after being involved in a terrible air accident.',
            'suffers from PTSD after being involved in a terrible accident in Disneyland.',
            'has been struggling with self-harm and feels like she is losing control of her impulses.',
            'has been experiencing symptoms of postpartum depression and is having trouble bonding with her newborn.',
            'has been struggling with work addiction and feels like he cannot take a break.',
            'has been struggling with trust issues and is having trouble opening up to others.',
            'has been suffering a lot since his girlfriend betrayed him two years ago.',
            'has been struggling with gambling addiction and feels like he is losing control of his finances.',
            'has been struggling with substance abuse for a long time.',
            'has been struggling with social isolation and feels like he is alone in the world.',
            'has been struggling with sexual dysfunction and is feeling anxious about intimacy.',
            'has been struggling with PTSD after serving in the military and is having trouble adjusting to civilian life.',
            'has been struggling with academic pressure and feels like she is constantly under stress.',
            'has been struggling with sexual identity issues and is feeling confused and scared.',
            'has been struggling with family issues and feels like he is caught in the middle of conflicts between his parents.',
            'has been struggling with school-related stress and is having trouble managing her workload.',
            'has been feeling overwhelmed and stressed as a single mother of three children.',
            'has been struggling with substance abuse and is having trouble managing her addiction as a single mother.',
            ]
therapists = ['is familiar with Psychodynamic Therapy and provides mental support to the patient.',
                'is familiar with Cognitive therapy and provides mental support to the patient.',
                'is familiar with Behavior therapy and provides mental support to the patient.',
                'is familiar with Humanistic therapy and provides mental support to the patient.',
                'asks questions and listen patitenly to their patients.',
                'cares about their patients very much.',
                'always comes up with good ideas to solve the problems of their patients.',
                'is professional at handling teenage problems.',
                'guides their patients to find solutions by asking them questions.',
                'never judges their patients.',
                'is familiar with Family Therapy and provides mental support to the patient.',
                'is familiar with Interpersonal Therapy and provides mental support to the patient.',
                'is able to build a trusting and supportive relationship with their clients, especially with retired people.']


def call_openai(messages, model):
    response = openai.ChatCompletion.create(
        model=model,
        temperature=0.8,
        max_tokens=3600,
        messages=messages,
    )
    return response


def run_it(messages, count):
    openai_response = call_openai(messages, model='gpt-3.5-turbo')
    ai_message = openai_response["choices"][0]["message"]["content"]
    file_name = f'data/therapy_{count}.txt'
    f = open(file_name, "w")
    f.write(ai_message)
    f.close()
    print(f'{count} generated...' )


def generation_dialogue():
    # print(len(patients), len(therapists))
    # return
    freeze_support()
    count = 1
    my_args = []
    for p in patients:
        for t in therapists:
            if count <= 240:
                count += 1
                continue
            if count > 320:
                return
            messages=[
                {"role": "system", "content": f"Generate a long dialogue with more than 600 words between a patient and a therapist. The patient {p}. The therapist {t}"},
            ]
            my_args.append((messages, count))
            if len(my_args) == 8:
                with Pool(8) as p:
                    print(p.starmap(run_it, my_args))
                my_args = []
            count += 1


def generate_json():
    total = 1035
    instructions = []
    format = 1
    for i in range(1, total+1):
        file_name = f'data/therapy_{i}.txt'
        f = open(file_name, "r")
        raw = f.read().replace('\n\n', '')
        sections = raw.split("Patient: ")
        for section in sections:
            if section == '':
                continue
            pair = section.split("Therapist: ")
            if len(pair) != 2:
                continue
            s0 = re.sub(r'\(.*\)', '', pair[0])
            s1 = re.sub(r'\(.*\)', '', pair[1])
            if s0 == '' or s1 == '':
                continue

            if format == 0:
                instructions.append({'instruction': s0, 'input': '', 'output': s1})
            if format == 1:
                instruction = """
                Below is a piece of dialogue between a patient and a therapist.
                The patient's words are in the Input.
                Write a response which is an appropriate reply to the patient from the therapist.
                """
                instructions.append({'instruction': instruction, 'input': s0, 'output': s1})
            if format == 2:
                instruction = """
                The Input is a dialogue between a patient and a therapist.
                Write a response which is an appropriate reply from the therapist based on the existing dialogue.
                """
                instructions.append({'instruction': instruction, 'input': s0, 'output': s1})

    my_len = len(instructions)
    print(my_len)
    f = open(f'data/therapy_no_input_{my_len}.json', "w")
    f.write(json.dumps(instructions, indent=4))
    f.close()


def generate_json_2_clean():
    for i in range(1, 1035+1):
        file_name = f'data/therapy_{i}.txt'
        f = open(file_name, "r")
        s = f.read()
        f.close()
        if '\n\n' not in s:
            s = s.replace('\n', '\n\n')
            f = open(file_name, "w")
            f.write(s)
            f.close()

    
def generate_json_2():
    total = 1035
    instructions = []
    my_ins = 'Below is a dialogue between a patient and a therapist. Write one reply as if you were the therapist.'
    for i in range(1, total+1):
        file_name = f'data/therapy_{i}.txt'
        f = open(file_name, "r")
        raws = f.read().split('\n\n')
        if raws[0].startswith('Therapist'):
            raws = raws[1:]
        n = len(raws)
        pt = 0
        while True:
            if n == 1:
                print(i, raws)
                break
            my_input = '\n\n'.join(raws[:pt+1])
            my_output = raws[pt+1]
            instructions.append({'instruction': my_ins, 'input': my_input, 'output': my_output})
            pt += 2
            if pt >= n-1:
                break

    f = open(f'data/therapy_no_input_{10}_1.json', "w")
    f.write(json.dumps(instructions, indent=4))
    f.close()


if __name__=="__main__":
    freeze_support()
    print('started...')
    # generation_dialogue()
    generate_json_2()
    print('ended...')