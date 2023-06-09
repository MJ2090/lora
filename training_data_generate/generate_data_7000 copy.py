import openai
import json
import re
from multiprocessing import Pool, freeze_support

ages = ['are in their late 30s.',
        'are a teenage.',
        'are a middle-aged person.',
        'in their early 50s',
        'are at late 60s.',]
jobs = ['were a college teacher.',
        'worked in tech company as an engineer.',
        'were a nurse in the hospital.',
        'worked in a Chinese restaurant.',
        'had a temporary job in a college as a cleaner.',
        'were a successful sales person in a startup.']
patients = ['recently lost their mother to a sudden illness. They are struggling to cope with the grief and feels overwhelmed by the responsibilities of managing the family estate. They are experiencing mood swings, difficulty concentrating, and have started to isolate themselves from friends.',
            'were recently promoted at work, and although they are excited about the new opportunity, they are plagued by self-doubt and anxiety about their ability to perform well in the new role. they have difficulty sleeping and have started experiencing panic attacks before important meetings.',
            'were involved in a serious car accident that left their physically and emotionally scarred. They experiences nightmares and flashbacks and avoids driving at all costs. They seeks therapy to help them process the trauma and regain control over Their life.',
            'are unhappy with their appearance, and it is affecting their self-esteem and mental health. They compares themselves to others on social media and struggles with disordered eating patterns. they feels hopeless about their ability to change and desires professional support.',
            'are experiencing severe mood swings, feelings of hopelessness, and difficulty bonding with their newborn baby. They struggles with guilt about their inability to feel happy and fears they are a bad mother. They seeks therapy to understand their feelings and find ways to cope.',
            'are struggling to maintain intimacy and communication in their relationship. They argue frequently and feel disconnected from one another. They seek therapy to help them understand and address their issues in order to rebuild their relationship.',
            'have been diagnosed with multiple sclerosis and struggles with depression and anxiety as a result. They are unsure of how to navigate their new reality and is overwhelmed by the uncertainty of their future. They seek therapy to develop coping strategies and build resilience.',
            'are seeking support in their journey to sobriety after a long battle with alcohol addiction. They feels ashamed of their past actions and fears relapse. They seeks therapy to develop healthy coping mechanisms and address the underlying issues that contributed to their addiction.',
            'are questioning their sexual identity and feel confused and isolated. They fear rejection from family and friends and struggle with feelings of shame and self-doubt. They seek therapy to explore their identity and learn how to navigate relationships and societal expectations.',
            'are experiencing ongoing harassment and bullying from their coworkers. They feel trapped, anxious, and humiliated, and their performance at work is suffering as a result. They seeks therapy to develop coping strategies and build self-esteem to better handle their situation at work.',
            'have a hard time making friends and participating in social events due to overwhelming anxiety. They often avoid social situations, which leaves them feeling lonely and isolated. They seek therapy to learn coping mechanisms and improve their social skills.',
            'have recently decided to change careers after being unsatisfied with their current job. They feel uncertain about the future and are experiencing stress and self-doubt during the transition. They seek therapy to help manage these emotions and navigate the challenges of starting a new career path.',
            'have recently experienced a miscarriage. They are struggling with feelings of grief, guilt, and helplessness, and it is affecting their mental health and relationships. They seek therapy to process their emotions and find support during this difficult time.',
            'have difficulty controlling their anger, which has led to problems in their personal and professional relationships. They feel ashamed of their outbursts and want to learn healthier ways to express their emotions. They seek therapy to develop anger management strategies.',
            'have recently seen their last child move out of the house. They are experiencing feelings of loneliness, loss of purpose, and sadness as they adjust to this new stage in life. They seek therapy to help them redefine their identity and explore new interests.',
            'have become the primary caregiver for their spouse who is suffering from Alzheimer disease. They feel overwhelmed, anxious, and drained as they struggle to balance their own needs with the demands of caregiving. They seek therapy to develop coping strategies and find support in their new role.',
            'struggle with perfectionism, which negatively impacts their mental health, relationships, and academic performance. They often feel stressed, overwhelmed, and unable to enjoy their accomplishments. They seek therapy to help them develop a healthier mindset and break the cycle of perfectionism.',
            'have a sibling who has been diagnosed with schizophrenia. They feel a mix of emotions, including fear, guilt, and a sense of responsibility for their sibling well-being. They seek therapy to better understand their sibling condition and learn how to support them while maintaining their own mental health.',
            'are in a long-distance relationship with their partner. They struggle with feelings of loneliness, jealousy, and insecurity as they navigate the challenges of maintaining a connection across the miles. They seek therapy to help them develop strategies for maintaining a healthy relationship despite the distance.',
            'are college student and are feeling overwhelmed by the demands of their coursework, extracurricular activities, and social life. They experience anxiety, difficulty sleeping, and a fear of failure, which is affecting their overall well-being. They seek therapy to learn strategies for managing stress and finding a healthy balance in their life.',
            'are feeling discontented with their life choices, career, and relationships. They question their purpose and struggle with feelings of regret and uncertainty. They seek therapy to explore their emotions, gain clarity, and find new meaning in their life.',
            'have a debilitating fear of flying that prevents them from traveling and affects their personal and professional life. They experience intense anxiety, panic attacks, and avoidance behavior. They seek therapy to confront their phobia and learn techniques to manage their anxiety.',
            'are an international student who recently moved to a new country for their studies. They struggle with feelings of loneliness, isolation, and homesickness, which negatively impact their academic performance and mental health. They seek therapy to adjust to their new environment and develop coping strategies for homesickness.',
            'have a long-term friend who has become increasingly manipulative and emotionally draining. They feel trapped in the friendship, experiencing guilt, resentment, and confusion. They seek therapy to learn how to set boundaries, assert their needs, and determine if the friendship is salvageable.',
            'are a successful professional who constantly feels inadequate and fears being exposed as a fraud, despite their accomplishments. They struggle with self-doubt, anxiety, and an inability to enjoy their success. They seek therapy to overcome impostor syndrome and build self-confidence.',
            'experienced physical, emotional, or sexual abuse during their childhood and continue to struggle with the long-term effects, such as low self-esteem, trust issues, and difficulty forming healthy relationships. They seek therapy to process their past trauma and build a healthier future.',
            'are a working parent struggling to manage the demands of their career and family life. They feel overwhelmed, guilty, and unable to fully enjoy either aspect of their life. They seek therapy to learn strategies for achieving a healthier work-life balance and reducing stress.',
            'have a child with autism and feel overwhelmed by the unique challenges of raising a child with special needs. They experience stress, frustration, and fear for their child future. They seek therapy to gain resources, coping strategies, and emotional support.',
            'suffer from chronic pain due to a medical condition, which affects their daily life, relationships, and mental health. They struggle with feelings of depression, hopelessness, and isolation as they try to manage their pain. They seek therapy to develop coping skills and improve their quality of life.',
            'are in a codependent relationship where they rely heavily on their partner for emotional support and validation. They struggle with low self-esteem, a fear of abandonment, and difficulty asserting their needs. They seek therapy to address codependency, establish boundaries, and cultivate a healthier sense of self-worth.',
            'are a professional who is required to give presentations at work, but they struggle with an intense fear of public speaking. They experience anxiety, sweaty palms, and a racing heart, which affects their performance. They seek therapy to build confidence and develop strategies for managing their fear.',
            'recently went through a painful breakup that left them feeling heartbroken, confused, and unsure of their future. They struggle with feelings of sadness, anger, and loneliness. They seek therapy to process their emotions, rebuild their self-esteem, and eventually move forward.',
            'are a high-achieving individual who has been pushing themselves to their limits in their career, leading to burnout. They feel exhausted, stressed, and disillusioned with their work. They seek therapy to develop self-care strategies, set boundaries, and rediscover passion in their life.',
            'struggle with intrusive thoughts and compulsive behaviors that interfere with their daily life, relationships, and mental health. They experience anxiety, shame, and a loss of control. They seek therapy to understand their OCD and develop coping mechanisms for managing their symptoms.',
            'have a boss who is overly critical, demanding, and unsupportive, which negatively impacts their job satisfaction and mental health. They feel stressed, anxious, and demotivated at work. They seek therapy to develop strategies for dealing with their boss and improving their work environment.',
            'are struggling to conceive a child, leading to feelings of grief, frustration, and inadequacy. Their fertility issues are straining their relationship with their partner and affecting their mental health. They seek therapy to process their emotions and explore alternative options for building a family.',
            'find themselves constantly checking their social media accounts, leading to feelings of anxiety, depression, and disconnection from reality. Their addiction is affecting their relationships, self-esteem, and overall well-being. They seek therapy to address their addiction and develop healthier habits.',
            'recently retired after a long career and are struggling to adjust to their new lifestyle. They feel a loss of purpose, boredom, and depression as they navigate this major life transition. They seek therapy to explore new interests, build social connections, and find meaning in their retirement years.',
            'have made past mistakes that they deeply regret, causing feelings of guilt, shame, and self-loathing. These emotions are affecting their mental health and preventing them from moving forward in their life. They seek therapy to process their feelings and learn how to forgive themselves.',
            'are questioning aspects of their identity, such as their cultural, racial, or religious background. They feel confused, isolated, and unsure of where they fit in. They seek therapy to explore their identity, find community, and develop a greater sense of self-understanding.',
            'recently experienced a significant life event, such as a move, job loss, or diagnosis of a serious illness. They feel overwhelmed, anxious, and uncertain about the future. They seek therapy to develop coping strategies and adapt to their new circumstances.',
            'struggle with trust issues in their relationships, stemming from past experiences of betrayal or abandonment. They have difficulty forming close connections and often push people away. They seek therapy to understand the roots of their trust issues and learn how to build healthy relationships.',
            'are dealing with significant financial difficulties, such as debt or job loss, which is causing stress, anxiety, and feelings of hopelessness. They seek therapy to develop coping strategies, explore practical solutions, and improve their emotional well-being.',
            'struggle with expressing themselves effectively, leading to misunderstandings and conflicts in their personal and professional relationships. They seek therapy to develop better communication skills and learn how to navigate difficult conversations.',
            'have a pattern of self-sabotaging their own success, whether it is in relationships, work, or personal goals. They struggle with feelings of unworthiness and fear of failure. They seek therapy to understand the reasons behind their self-sabotage and learn strategies for overcoming it.',
            'recently lost a beloved pet which was very important to the family, leaving them feeling devastated, lonely, and heartbroken. The pet has been with them for a long time. They seek therapy to process their grief, honor their pet memory, and find ways to heal.',
            'are a single parent facing the emotional and practical challenges of raising a child on their own. They feel overwhelmed, isolated, and worried about their ability to provide for their child needs. They seek therapy for support, guidance, and coping strategies.',
            'have experienced bullying, whether in school or the workplace, which has led to feelings of depression, anxiety, and low self-esteem. They seek therapy to address the emotional impact of bullying and develop strategies for standing up for themselves and building resilience.',
            'have difficulty committing to long-term relationships, jobs, or other major life decisions. They fear making the wrong choice or feeling trapped in a situation. They seek therapy to explore their fear of commitment and develop strategies for overcoming it.',
            'struggle with the pressure to conform to the expectations and behaviors of their peers, leading to feelings of stress, anxiety, and confusion about their own values. They seek therapy to build self-confidence, assert their individuality, and learn how to resist peer pressure.',
            'experience intense anxiety when faced with exams or other high-stakes evaluations, which can negatively impact their performance. They seek therapy to learn techniques for managing test anxiety and improving their performance under pressure.',
            'are a highly sensitive person who is easily overwhelmed by sensory input, emotions, and the demands of daily life. They struggle with feelings of anxiety, exhaustion, and isolation. They seek therapy to understand their sensitivity, develop coping strategies, and build.',
            'struggle with negative perceptions of their body, which affects their self-esteem, relationships, and overall well-being. They seek therapy to address the root causes of their body image issues and develop a healthier relationship with their body.',
            'have been diagnosed with a chronic illness, such as diabetes or fibromyalgia, which significantly impacts their daily life and mental health. They experience feelings of depression, anxiety, and helplessness. They seek therapy to learn coping strategies and find support in managing their illness.',
            'have difficulty navigating conflicts in their personal and professional relationships, often avoiding confrontation or engaging in unhealthy patterns of communication. They seek therapy to develop effective conflict resolution skills and strengthen their relationships.',
            'are part of a blended family and face challenges related to step-parenting, sibling relationships, and navigating family dynamics. They seek therapy to find support and guidance in creating a harmonious family environment.',
            'are questioning their gender identity and exploring the possibility that they may be transgender, non-binary, or genderqueer. They seek therapy to gain a better understanding of their identity, find support, and navigate potential challenges related to transitioning or coming out.',
            'recently experienced a traumatic event, such as a natural disaster, serious accident, or violent crime, which has left them feeling anxious, fearful, and overwhelmed. They seek therapy to process the trauma, develop coping strategies, and regain a sense of safety and control in their life.',
            'are experiencing sexual difficulties, such as low libido, erectile dysfunction, or painful intercourse, which negatively impact their relationships and self-esteem. They seek therapy to explore the underlying causes of these issues and develop strategies for improving their sexual health.',
            'are experiencing high levels of stress at work due to factors such as a demanding workload, office politics, or a lack of job security. They seek therapy to develop stress management techniques, improve their work-life balance, and build resilience in the face of workplace challenges.',
            'struggle with addiction, whether to substances or behaviors, which negatively impacts their physical health, relationships, and overall well-being. They seek therapy to address the root causes of their addiction and develop strategies for achieving and maintaining sobriety.',
            'have a loved one who has been diagnosed with a terminal illness, and they are struggling with feelings of grief, helplessness, and anticipatory loss. They seek therapy to process their emotions, find support, and learn how to care for their loved one during this difficult time.',
            'have children who have recently left home, and they are experiencing feelings of sadness, loneliness, and a loss of purpose. They seek therapy to process their emotions, explore new interests, and adjust to this new phase in their life.',
            'struggle with perfectionism, which leads to unrealistic expectations, chronic stress, and a constant feeling of inadequacy. They seek therapy to understand the roots of their perfectionism and develop strategies for embracing self-compassion and a more balanced approach to life.',
            'are experiencing bullying or harassment at work, which has led to feelings of anxiety, depression, and decreased job satisfaction. They seek therapy to address the emotional impact of the bullying, develop assertiveness skills, and explore potential solutions or coping strategies.',
            'experience intense feelings of anxiety and distress when separated from a loved one, which impacts their relationships and well-being. They seek therapy to understand the underlying causes of their separation anxiety and develop coping mechanisms for managing their symptoms.',
            'are in a long-distance relationship and struggle with feelings of loneliness, insecurity, and maintaining emotional intimacy with their partner. They seek therapy to develop strategies for maintaining a healthy and satisfying long-distance relationship.',
            'suffer from chronic insomnia, which negatively affects their mood, energy levels, and overall quality of life. They seek therapy to identify the underlying causes of their sleep problems and develop techniques for improving their sleep hygiene and relaxation.',
            'are a highly intelligent or gifted individual who struggles with feelings of isolation, imposter syndrome, and difficulty forming meaningful connections with others. They seek therapy to better understand their unique identity, develop coping strategies, and build healthy relationships.',
            'experience ongoing conflict and competition with a sibling, which has strained their relationship and caused emotional distress. They seek therapy to address the roots of the rivalry, improve communication, and strengthen their sibling bond.',
            'are considering a significant career change, which has left them feeling uncertain, anxious, and overwhelmed by the decision-making process. They seek therapy to explore their values, skills, and goals, and to develop a plan for a successful career transition.',
            'struggle with a fear of intimacy, which affects their ability to form deep, meaningful relationships. They have difficulty trusting others, opening up emotionally, and expressing vulnerability. They seek therapy to address the root causes of their fear and develop strategies for building intimacy in their relationships.',
            'have developed a gambling addiction that started as a casual pastime but has gradually escalated. They have spent large sums of money, leading to financial instability, debt, and even the loss of their job. Their relationships with family and friends have deteriorated as they prioritize gambling over other aspects of their life. They seek therapy to understand the underlying issues driving their addiction, develop coping strategies, and rebuild their life.',
            'recently gave birth and are struggling with postpartum depression, which is negatively impacting their ability to bond with their newborn, as well as their relationships with their partner and other family members. They experience feelings of sadness, guilt, hopelessness, and exhaustion, and may have difficulty completing everyday tasks. They seek therapy to address the emotional and hormonal factors contributing to their postpartum depression and develop strategies for self-care and support.',
            'have sustained a traumatic brain injury in an accident and are dealing with the physical, cognitive, and emotional challenges that follow. They may experience memory issues, difficulty concentrating, mood swings, and changes in their personality. They seek therapy to help them adapt to these changes, regain a sense of control, and rebuild their identity and self-esteem.',
            'suffer from trichotillomania, a compulsive hair-pulling disorder that has led to noticeable hair loss and feelings of shame and embarrassment. They struggle to control their urges to pull their hair, which negatively affects their self-esteem, relationships, and overall well-being. They seek therapy to understand the psychological factors driving their behavior, develop coping mechanisms, and address the emotional impact of their condition.',
            'are experiencing a midlife crisis, which has led to feelings of dissatisfaction, restlessness, and a desire for significant life changes. They may question their career, relationships, and overall life choices, leading to impulsive decisions and emotional turmoil. They seek therapy to explore their feelings, identify their values and goals, and develop strategies for navigating this transitional period with clarity and purpose.',
            'have been diagnosed with an eating disorder, such as anorexia nervosa, bulimia nervosa, or binge eating disorder. They struggle with their relationship to food, body image, and self-worth, which has led to dangerous eating habits and significant health consequences. They seek therapy to address the psychological factors contributing to their eating disorder, develop healthier coping strategies, and rebuild their relationship with food and their body.',
            'have recently experienced a significant life change, such as the death of a loved one, a breakup, or a job loss, which has led to an adjustment disorder. They experience symptoms of anxiety, depression, and difficulty functioning in their daily life. They seek therapy to help them process their emotions, adapt to their new circumstances, and develop coping strategies for moving forward.',
            'are in a relationship with a narcissistic partner who consistently manipulates, gaslights, and undermines their self-esteem. They struggle to assert their needs and boundaries, and often feel emotionally drained and isolated. They seek therapy to understand the dynamics of their relationship, develop assertiveness and self-worth, and explore potential solutions or coping strategies.',
            'have recently been diagnosed with or acquired a physical disability that has significantly altered their daily life and self-perception. They may experience feelings of grief, anger, and frustration as they adapt to their new circumstances and confront societal stigmas. They seek therapy to process their emotions.',
            'have developed a hoarding disorder, which has resulted in an accumulation of clutter, disorganization, and unsanitary living conditions in their home. They have difficulty discarding items, even those with little or no value, leading to strained relationships with family and friends. They seek therapy to understand the emotional and psychological factors contributing to their hoarding behaviors, develop coping strategies, and begin the process of decluttering and healing.',
            'find themselves in a pattern of codependent relationships, in which they excessively rely on their partner for emotional support and validation. This often leads to feelings of resentment, loss of individuality, and an imbalance of power within the relationship. They seek therapy to understand the root causes of their codependency, develop healthier communication and boundary-setting skills, and foster more balanced relationships.',
            'are experiencing burnout due to chronic workplace stress, an unmanageable workload, or a lack of support from colleagues or supervisors. They feel exhausted, demotivated, and emotionally drained, which has led to a decline in their job performance and overall well-being. They seek therapy to identify the causes of their burnout, develop strategies for self-care and stress management, and explore potential career changes or workplace solutions.',
            'have been diagnosed with social anxiety disorder, which causes them to experience intense fear and anxiety in social situations. This has led to avoidance of social activities, strained relationships, and feelings of isolation and loneliness. They seek therapy to address the underlying causes of their social anxiety, develop coping mechanisms, and improve their social skills and self-confidence.',
            'experienced significant trauma during their childhood, such as abuse, neglect, or witnessing violence. These events continue to impact their mental health, relationships, and overall well-being. They seek therapy to process their past experiences, address unresolved emotions, and develop strategies for healing and growth.',
            'suffer from panic disorder, which is characterized by recurrent panic attacks and a constant fear of experiencing another attack. This has led to avoidance of certain situations, impaired functioning, and decreased quality of life. They seek therapy to understand the triggers and underlying causes of their panic attacks, learn coping strategies, and address the emotional impact of their disorder.',
            'have been trying to conceive a child without success, which has led to feelings of grief, inadequacy, and strain in their relationship with their partner. They seek therapy to process their emotions, explore alternative options for parenthood, and find support in navigating the challenges of infertility.',
            'have been diagnosed with OCD, which is characterized by intrusive thoughts and compulsive behaviors that significantly impact their daily life. They may engage in repetitive rituals, such as excessive hand washing or checking, to alleviate their anxiety. They seek therapy to address the cognitive and behavioral aspects of their OCD, develop coping strategies, and improve their overall quality of life.',
            'have aphantasia, a condition in which they are unable to create mental images or visualize things in their mind eye. This can lead to difficulties with memory, creativity, and emotional processing. They seek therapy to better understand their unique cognitive experience, develop alternative strategies for processing information, and find support in adapting to the challenges of living with aphantasia.',
            'have been diagnosed with adult attention-deficit/hyperactivity disorder (ADHD), which impacts their ability to focus, organize, and complete tasks. They may experience difficulties at work, strained relationships, and feelings of frustration or inadequacy. They seek therapy to develop coping strategies, improve time management and organizational skills, and address the emotional impact of living with ADHD.',
            'have been a target of cyberbullying, which has led to feelings of fear, anxiety, and depression. The harassment may have damaged their reputation, affected their relationships, or caused them to withdraw from social media and online activities. They seek therapy to address the emotional impact of cyberbullying, develop coping strategies, and rebuild their self-esteem.',
            'recently lost a beloved pet, which has led to feelings of grief, sadness, and guilt. Their pet was an important part of their life, and they are struggling to cope with the loss and adjust to a new routine without their companion. They seek therapy to process their grief, find support, and navigate the healing process.',
            'engage in self-harm behaviors, such as cutting or burning themselves, as a way to cope with emotional pain or distress. This behavior puts them at risk for serious injury and has negative effects on their mental health, self-esteem, and relationships. They seek therapy to understand the underlying reasons for their self-harm, develop healthier coping mechanisms, and address the emotional impact of their actions.',
            'suffer from impostor syndrome, which is characterized by feelings of inadequacy, self-doubt, and a fear of being exposed as a fraud, despite evidence of their achievements and competence. This can lead to anxiety, depression, and a reluctance to pursue new opportunities. They seek therapy to address the underlying causes of their impostor syndrome, build self-confidence, and develop strategies for overcoming self-doubt.',
            'have a specific phobia, such as a fear of flying, heights, or spiders, which significantly impacts their daily life and limits their ability to participate in certain activities. They experience intense anxiety and distress when confronted with the object or situation they fear. They seek therapy to understand the root causes of their phobia, develop coping strategies, and work towards overcoming their fear.',
            'struggle with feelings of jealousy and envy, which negatively affect their relationships, self-esteem, and overall well-being. They may compare themselves to others, harbor resentment, or engage in unhealthy behaviors as a result of these emotions. They seek therapy to address the underlying causes of their jealousy and envy, develop strategies for managing their emotions, and improve their overall emotional intelligence.',
            'are experiencing a quarter-life crisis, marked by feelings of uncertainty, dissatisfaction, and confusion about their life direction. They may question their career path, relationships, and overall life choices, leading to emotional turmoil and a sense of being lost. They seek therapy to explore their feelings, identify their values and goals, and develop strategies for navigating this transitional period with clarity and purpose.',
            'have a paralyzing fear of failure, which prevents them from taking risks, pursuing new opportunities, or stepping outside of their comfort zone. This fear may be rooted in perfectionism, past experiences, or societal expectations. They seek therapy to understand the origins of their fear, develop strategies for embracing risk and resilience, and address the emotional impact of their fear of failure.',
            'struggle with performance anxiety, which causes them to experience intense fear and nervousness when speaking or performing in public. This anxiety may lead to avoidance of such situations, missed opportunities, and feelings of shame or embarrassment. They seek therapy to understand the root causes of their performance anxiety, develop coping strategies, and improve their self-confidence in public settings.',
            'suffer from test anxiety, which negatively impacts their academic performance and overall educational experience. They may experience physical symptoms, such as headaches or nausea, and cognitive symptoms, such as difficulty concentrating or recalling information during exams. They seek therapy to address the underlying causes of their test anxiety, develop relaxation techniques, and improve their test-taking skills.',
            'have recently moved away from home for college, work, or another reason, and are struggling with feelings of homesickness, loneliness, and difficulty adjusting to their new environment. They seek therapy to process their emotions, develop coping strategies for managing their homesickness, and build a support system in their new location.',
            'have been diagnosed with a chronic illness, such as diabetes or multiple sclerosis, which has significantly impacted their daily life and emotional well-being. They may experience feelings of grief, anger, and frustration as they adjust to their new reality and confront the challenges of managing their condition. They seek therapy to process their emotions, develop coping strategies, and build resilience.',
            'suffer from body dysmorphic disorder, which is characterized by an excessive preoccupation with a perceived flaw or defect in their appearance. This preoccupation leads to feelings of shame, depression, and anxiety, and may result in social withdrawal or other unhealthy behaviors. They seek therapy to address the underlying causes of their BDD, develop healthier body image perceptions, and improve their overall mental health.',
            'are experiencing ongoing conflicts within their romantic relationship or marriage, which have led to feelings of dissatisfaction, resentment, and emotional distance. They seek therapy, either individually or as a couple, to address the roots of their relationship issues, improve communication, and strengthen their emotional connection.',
            'have been diagnosed with seasonal affective disorder, a type of depression that typically occurs during the winter months. They experience symptoms such as fatigue, irritability, and a loss of interest in activities they once enjoyed. They seek therapy to address the underlying causes of their SAD, develop strategies for managing their symptoms, and improve their overall well-being.',
            'are a caregiver for a family member or loved one who has a chronic illness or disability, and they are struggling with feelings of stress, burnout, and guilt. Their caregiving role has taken a toll on their mental health, relationships, and personal life. They seek therapy to process their emotions, develop strategies for self-care and stress management, and find support in navigating the challenges of caregiving.',
            ]
therapists = ['is familiar with Cognitive behavioral therapy, asks questions based on the conversation to and provides professional mental support to the patient.',
                'often guides their patients through their way to find solutions by asking them questions based on their background and experiments.',
                'cares about their patients, ask patients questions, and is familiar with Family Therapy, providing mental support to patients of all ages.',
                'is able to build a trusting and supportive relationship with their clients, especially with middle-aged and retired people.']


def call_openai(messages, model):
    response = openai.ChatCompletion.create(
        model=model,
        temperature=0.85,
        max_tokens=3700,
        messages=messages,
    )
    return response


def run_it(messages, count):
    print(f'{count} started...' )
    openai_response = call_openai(messages, model='gpt-3.5-turbo')
    ai_message = openai_response["choices"][0]["message"]["content"]
    file_name = f'data_1/therapy_1_{count}.txt'
    f = open(file_name, "w")
    f.write(ai_message)
    f.close()
    print(f'{count} generated...' )


def generation_dialogue():
    # print(len(patients), len(therapists))
    # return
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
                {"role": "system", "content": f"Generate a long dialogue with more than 800 words between a patient and a therapist. The patient {p}. The therapist {t}"},
            ]
            my_args.append((messages, count))
            if len(my_args) == 8:
                with Pool(8) as p:
                    print(p.starmap(run_it, my_args))
                my_args = []
            count += 1


def generation_dialogue_new():
    # print(len(patients), len(therapists))
    # return
    print('oook')
    count = 1
    my_args = []
    for a in ages:
        for j in jobs:
            for p in patients:
                for t in therapists:
                    if count <= 146:
                        count += 1
                        continue
                    messages=[
                        {"role": "system", "content": f"Generate a long conversation with more than 2000 words between a Patient and a Therapist. The Patient {a}, and {j}. The Patient {p}. The Therapist {t}"},
                    ]
                    my_args.append((messages, count))
                    if len(my_args) == 6:
                        with Pool(6) as p:
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
    generation_dialogue_new()
    # generate_json_2()
    print('ended...')