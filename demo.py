import re
import torch
from PIL import Image
from openai import OpenAI
from transformers import LlavaForConditionalGeneration, AutoProcessor

Deepseek_API_KEY = "Your_Deepseek_API_KEY"
Deepseek_API_BASE = "Your_Deepseek_API_BASE"

out_print = True
modules = ["classifier", "visualist", "captor", "analyst"]

def mllm_get_response(user_input, image):
    model_id = "llava-hf/llava-1.5-7b-hf"
    device = 0
    model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    temperature = 0.0
    max_tokens = 2560

    prompt = "USER: <image>\n" + user_input + "\nASSISTANT:"
    inputs = processor(prompt, image, return_tensors='pt').to(device, torch.float16)
    output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    output = processor.decode(output[0][2:], skip_special_tokens=True)
    output = output[output.rfind("ASSISTANT: ") + len("ASSISTANT: "):]
    return output

def llm_get_response(user_input, system_input=""):
    deepseek_api_key = Deepseek_API_KEY
    assert deepseek_api_key is not None
    deepseek_api_base = Deepseek_API_BASE
    assert deepseek_api_base is not None
    model_name = "deepseek-chat"
    client = OpenAI(
        api_key=deepseek_api_key,
        base_url=deepseek_api_base)
    response = None
    i = 0
    message = [{'role': 'system', 'content': system_input},
                {'role': 'user', 'content': user_input}]
    while i < 3:
        response = client.chat.completions.create(
            model=model_name,
            messages=message,
            temperature=0.0,
            seed=127,
            max_tokens=1000)
        i += 1
        if response is not None and response.choices[0].message.content is not None:
            response = response.choices[0].message.content
            # print(response)
            break
        else:
            continue
    return response

def match_tasks(decision):
    pattern_classifier = (
        r"\[Classifier Module:\s*(.*?\.)\]|"
        r"\[Classifier Module:\s*(.*?)\n\]|"
        r"Classifier Module:\s+((?: *- .+?\n)+)|"
        r"Classifier Module:\s+(.+?)(?=\n2\.)|"
        r"Classifier Module\*{0,2}:\*{0,2}\s*(.*?)\n"
    )
    pattern_visualist = (
        r"\[Visualist Module:\s*(.*?\.)\]|"
        r"\[Visualist Module:\s*(.*?)\n\]|"
        r"Visualist Module:\s+((?: *- .+?\n)+)|"
        r"Visualist Module:\s+(.+?)(?=\n2\.)|"
        r"Visualist Module\*{0,2}:\*{0,2}\s*(.*?)\n"
    )
    pattern_captor = (
        r"\[Captor Module:\s*(.*?\.)\]|"
        r"\[Captor Module:\s*(.*?)\n\]|"
        r"Captor Module:\s+((?: *- .+?\n)+)|"
        r"Captor Module:\s+(.+?)(?=\n2\.)|"
        r"Captor Module\*{0,2}:\*{0,2}\s*(.*?)\n"
    )
    pattern_analyst = (
        r"\[Analyst Module:\s*(.*?\.)\]|" 
        r"\[Analyst Module:\s*(.*?)\n\]|" 
        r"Analyst Module:\s+((?: *- .+?\n)+)|"
        r"Analyst Module:\s+(.+?)(?=\n2\.)|"
        r"Analyst Module\*{0,2}:\*{0,2}\s*(.*?)\n"
)

    match_classifier = re.search(pattern_classifier, decision, re.DOTALL)
    match_visualist = re.search(pattern_visualist, decision, re.DOTALL)
    match_captor = re.search(pattern_captor, decision, re.DOTALL)
    match_analyst = re.search(pattern_analyst, decision, re.DOTALL)

    return {"classifier": match_classifier, "visualist": match_visualist, "captor": match_captor, "analyst": match_analyst}

def module_required(text):
    if text != "N/A" and text != "None" and text.find("Not applicable") == -1 and text.find("Not required") == -1:
        return True
    return False

def reference_prompt():
    img = 'Image modality/type: color image, grayscale image, panchromatic image(a high-resolution grayscale image); aerial image, satellite image; low-resolution image, high-resolution image.'
    obj = 'The theme/scene of image: bare land, church, commercial, pond, park, railway station, square, viaduct, parking lot, sparse/medium/dense residential, etc.'
    pos = 'The absolute position of objects in the image/Where objects located in the image: center, upper part, lower part, left part, right part, top left corner, top right corner, bottom left corner, bottom right corner.'
    dir = 'Direction: south-north, west-east, falling, rising, etc.'
    traf = 'The comprehensive traffic situations: not important, important with 1 intersection, important with several bridges, etc.'
    suit = 'The water situations around the agricultural land: clean, polluted, no water, no agricultural land.'
    land = 'The land use types: agricultural areas, educational areas, commercial areas, park, residential areas, woodland areas, etc.'
    need = 'The needs for the renovation of village or residents: no needs, the roads need to be improved, the greening needs to be supplemented, urban villages need attention, etc.'
    road1 = 'The road types: no roads, wide lanes, one-way lanes, railways, etc.'
    road2 = 'The road materials: no roads, asphalt, cement, unsurfaced, etc.'
    return img + pos + obj + dir + road1 + road2 + traf + suit + land + need

# Dynamic Selection
def prompt_dynamic_selector(question, dataset_name):
    sys = (
        f"You're an expert in judging the difficulty of questions. Your task is to judge the difficulty of questions in {dataset_name}."
        f"You need to determine the difficulty of the question, with 0 indicating simplicity and 1 indicating difficulty, according to the following criteria:\n"
        f"0: The question is simple that needs to be judged based on factual analysis (existence, quantity, color, orientation, scene, etc.).\n"
        f"1: The question is difficult that requires multi-step situation analysis to be accurately answered, and factual analysis is only one part of it.\n"
        f"Remember, the question is either simple or difficult, when judging the question, do not introduce additional standards.\n")

    user = (
        f"Now, here is a question based on some remote sensing images: {question}.\n"
        f"Please analyze the difficulty of the question based on the criteria for difficulty determination, and output the number corresponding to the difficulty level (0 or 1).\n"
        f"Your output should include your reason and be in the exact same format as '''Result:[Your choice]'''")
    return sys, user

def get_gpt_response(sys, user, client):
    response = None
    i = 0
    message = [{'role': 'system', 'content': sys},
               {'role': 'user', 'content': user}]
    while i < 3:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=message,
            temperature=0,
            seed=127,
            max_tokens=1000)
        i += 1
        if response is not None and response.choices[0].message.content is not None:
            response = response.choices[0].message.content
            if response.rfind("Result:") == -1:
                response = None
                continue
            else:
                match = re.search(r"Result:\s*(\d)", response)
                if match:
                    response = match.group(1)
                    return response
                else:
                    response = None
                    continue
        else:
            response = -100
            continue
    return response

# Coarse-Grained
def get_base_answer_prompt(question, ref):
    prompt = (
        f"Question: {question}. "
        f"Please directly answer the question based on the given image and references.\n References: {ref}.\n."
    )
    return prompt

def get_caption_prompt():
    prompt = (
        "You are a knowledgeable and skilled information integration remote sensing expert.\n"
        f"Please provide detailed descriptions of the visable objects' features in the remote sensing image. "
        "Features include the theme, absolute position, object visibility, relative position, color and quantity, etc. "
        "The example for your reference is as follows: \n"
        '''This is a high-resolution aerial color image that shows an airport terminal building (theme). 
        A plane parked on the upper left corner of the image (absolute position), 
        and there is a road running east to west (direction), 
        with only the tip of the nose visible (object visibility). 
        The surrounding area is a parking lot (relative position), 
        with many vehicles of different sizes and shapes neatly parked on yellow (color) marked lines on the ground. 
        There are a total of 20 (quantity) large vehicles in the image.'''
    )
    return prompt

# Fine-grained    
def decision_making_prompt(question):
    prompt_expert = (
        "You are a advanced question-answering agent equipped with 4 specialized modules to aid in analyzing and responding to questions about remote sensing images:\n\n"
        "1. Analyst Module:\n"
        "Abilities:\n"
        "1) Identify the land use types in the scene of the image (e.g., agricultural, educational, commercial, park, residential, woodland areas).\n"
        "2) Determine the road materials (e.g., asphalt, cement, unsurfaced, no roads) and types (e.g., wide lanes, one-way lanes, railways, no roads) around the village or residential area.\n"
        "3) Analyze the water situations around the agricultural land (e.g., clean, polluted, no water, no agricultural land).\n"
        "4) Analyze the comprehensive traffic situations in the scene of the image (e.g., important with 1 intersection, important with several bridges, not important).\n"
        "5) Consider the needs for the renovation of village or residents (e.g., the roads need to be improved, the greening needs to be supplemented, urban villages need attention, no needs).\n"
        "When information from this module is needed, specify your request as: \"Analyst Module: <specific task or information to extract>.\"\n\n"
        "2. Visualist Module:\n"
        "Abilities:\n"
        "1) Identify the main scene (e.g., commerical, pond, port, square, viaduct).\n"
        f"2) Detect and count objects relevant to the question: '{question}' within an image.\n"
        "When you need this module, frame your request as: 'Visualist Module: <object1, object2, ..., objectN>,' listing the objects you believe need detection for further analysis.\n\n"
        "3. Captor Module:\n"
        "Abilities:\n"
        f"1) Pinpoint the absolute position of the object relevant to the question: '{question}' within the image (e.g., upper part/lower part/top left corner/center of the image).\n"
        f"2) Provide the relative position between objects relevant to the question: '{question}'.\n"
        "When information from this module is needed, specify your request as: \"Captor Module: <specific task or information to extract>.\"\n\n"
        "4. Classifier Module:\n"
        "Abilities:\n"
        "1) Determine the appropriate imaging modality (e.g., color or grayscale/panchromatic(a high-resolution grayscale image)).\n"
        "2) Identify the imaging height (aerial or satellite).\n"
        "3) Analyze the resolution level of remote sensing images (low or high).\n"
        "When this module is required, specify your request as: 'Classifier Module: <specific task or information to extract>.'\n\n"
    )
    prompt_form = (
        "When faced with a question about an image, which will be accompanied by a description that might not cover all its details, your task is to:\n\n"
        "If the question can be answered directly based on the information provided without the need for detailed input from the modules, specify this explicitly. Do not disclose the answer itself. \n"
        "Otherwise:\n"
        "- Provide a rationale for your approach to answering the question, explaining how you will use the information from the image and the modules to form a comprehensive answer.\n"
        "- Assign specific tasks to each module as needed, based on their capabilities, to gather additional information essential for answering the question accurately.\n\n"
        "Your response should be structured as follows:\n\n"
        "Answer: "
        "['This question does not require any modules and can be answered directly based on the information provided.'] or "
        "[Rationale: Your explanation of how you plan to approach the question, including any initial insights based on the question and image description provided. Explain how the modules' input will complement this information.]\n\n"
    )
    prompt_tasks = (
        "Modules' tasks  (if applicable):\n\n"
        "[Clearly list in detail the tasks that need to be completed by the classifier/visualist/captor/analyst module.]\n"
        "Ensure your response adheres to this format to systematically address the question using the available modules or direct analysis as appropriate.\n"
        f"Please refer to the prompts and examples above to help me solve the following problem:{question}\n"
    )

    return prompt_expert + prompt_form + prompt_tasks

def get_mllm_answer_prompt(task):
    mllm_prompt = f"Please give a detailed response to this task.\nTask: {task}.\n"
    return mllm_prompt

def get_integrate_answer_prompt(question, cls, vis, cap, ana, des, caption, ref):
    suplement = ""
    if cls is not None:
        suplement += f"Classifier Module: {cls}. "
    if vis is not None:
        suplement += f"Visualist Module: {vis}. "
    if cap is not None:
        suplement += f"Captor Module: {cap}. "
    if ana is not None:
        suplement += f"Analyst Module: {ana}. "
    prompt_pt1 = (
        "You are a knowledgeable and skilled information integration remote sensing expert. Please gradually think and answer the questions based on the given questions and supplementary information.\n"
        "Please note that we not only need answers, but more importantly, we need rationales for obtaining answers.\n"
        "Please do not answer with uncertainty, try your best to give a concise and clear answer.\n"
        f"Here is a description of the related remote sensing image: {caption}.\nReferences: {ref}\n"
        )
    if cls is None and vis is None and cap is None and ana is None:
        prompt = (f"{prompt_pt1}"
                  f"The expected response format is as follows: Rationale:<rationale> Answer:<answer>.\n"
                  f"Please answer the following case: Question: <{question}>, Supplementary information: {des}.\n"
                  )
    else:
        prompt = (f"{prompt_pt1}"
                  f"The expected response format is as follows: Rationale:<rationale> Answer:<answer>.\n"
                  f"Please answer the following case: Question: <{question}>, Supplementary information: {suplement}.\n"
                  )
    return prompt


# Main
ref = reference_prompt()
mllm_answer_dict = {}
dataset_name = "EarthVQA"
data = [{"image_name": "images_png/3759.png", "question": "What are the comprehensive traffic situations in this scene?", "answer": "This is a very important traffic area with 1 intersection, and several bridges"}]
img_path, question, answer = data[0]["image_name"], data[0]["question"], data[0]["answer"]
img = Image.open(img_path)
format_prompt_0 = "Answer the question using a single word or phrase. "
format_prompt_1 = "Answer the question using a single short sentence. "
sign = 0

sys, user = prompt_dynamic_selector(question, dataset_name)
deepseek_api_key = Deepseek_API_KEY
assert deepseek_api_key is not None
openai_api_base = Deepseek_API_BASE

if openai_api_base is not None:
    client = OpenAI(
        api_key=deepseek_api_key,
        base_url=openai_api_base
    )
else:
    client = OpenAI(
        api_key=deepseek_api_key,
    )
response = get_gpt_response(sys, user, client)
if response is None:
    sign = 1
else:
    if response == -100:
        raise ValueError("token limit reached")
    sign = int(response)


if sign:
    question = question[:-1] + "? " + format_prompt_1
    ans = mllm_get_response(get_base_answer_prompt(question, ref), img)
    if out_print:
        print(f"Base Answer: {ans}")
    caption = mllm_get_response(get_caption_prompt(), img)
    if out_print:
        print(f"Image Caption: {caption}")

    decision = llm_get_response(decision_making_prompt(question))
    decision += "\n"
    if out_print:
        print(f"Decision: {decision}")

    match_results = match_tasks(decision)
    for module in modules:
        match = match_results[module]
        if match:
            print("-" * 50)
            if match.group(1):
                task = match.group(1).strip()
            elif match.group(2):
                task = match.group(2).strip()
            elif match.group(3):
                task = match.group(3).strip()
            elif match.group(4):
                task = match.group(4).strip()
            else:
                task = match.group(5).strip()
            if module_required(task):
                print(task)
                print("-" * 50)

                mllm_answer = mllm_get_response(get_mllm_answer_prompt(task), img)
                if mllm_answer.find("Answer:") != -1:
                    mllm_answer = mllm_answer[mllm_answer.find("Answer:") + len("Answer:"):].strip()
                if mllm_answer == "":
                    mllm_answer = None
            else:
                mllm_answer = None
        else:
            task = None
            mllm_answer = None
        mllm_answer_dict[module] = mllm_answer
    if out_print:
        print(f"MLLM Answer Dict: {mllm_answer_dict}")

    integrate_answer = llm_get_response( 
        get_integrate_answer_prompt(question, mllm_answer_dict["classifier"],
                                    mllm_answer_dict["visualist"],
                                    mllm_answer_dict["captor"], 
                                    mllm_answer_dict["analyst"], ans, caption, ref))
    if out_print:
        print(f"Final Answer: {integrate_answer}")

else:
    question = question[:-1] + "? " + format_prompt_0
    ans = mllm_get_response(get_base_answer_prompt(question, ref), img)
    final_answer = ans
    if out_print:
        print(f"Final Answer: {final_answer}")
