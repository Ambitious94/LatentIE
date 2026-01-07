
def build_agent_message_sequential_latent_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert method in ["latent_mas"], "this prompt only for latent_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    if role == "planner":
        user_prompt = f"""You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

Question: {question}

Your outlined plan should be concise with a few bulletpoints for each step. Do not produce the final answer.
Now output your plan to solve the question below:
"""
    
    elif role == "critic":
        user_prompt = f"""
Question: {question}

You are a Critic Agent to evaluate the correctness of the input plan for the given question and provide helpful feedback for improving the plan.
The plan information is provided in latent KV representation format. Review the plan and question and output:
(1) original plan contents
(2) constructive feedback on the original plan.

Format your response as follows:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""
    
    elif role == "refiner":
        user_prompt = f"""
Question: {question}

You are a Refiner Agent to provide a refined step-by-step plan for solving the given question.
You are provided with:
(1) latent-format information: a previous plan with feedback
(2) text-format information: the input question you need to solve.

Based on the input, write a refined and improved plan to solve the question. Make sure your output plan is correct and concise.

Now, output your refined plan below:
"""
    
    elif role == "judger":
        if args.task in ['gsm8k', 'aime2024', 'aime2025']:
            user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""
        
        elif args.task in ["arc_easy", "arc_challenge", "gpqa", 'medqa']:
            user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif args.task in ["mbppplus", "humanevalplus"]:
            user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve.

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block.

Now, reason step by step and output the final answer inside ```python
YOUR_PYTHON_CODE
```.
"""

        elif args.task in ["winogrande"]:
            user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        else: 
            raise NotImplementedError(f"Task {args.task} not implemented in v5 judger prompt.")
        
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]


def build_agent_message_hierarchical_latent_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert method in ["latent_mas"], "this prompt only for latent_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    if args.task in ['gsm8k', 'aime2024', 'aime2025']:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""

    elif args.task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:

        if args.task == "medqa":

            if role == "planner":
                user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}

Your response:
"""
            elif role == "critic":
                user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}     

Your response:
"""
            elif role == "refiner":
                user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}

Your response:       
"""
            elif role == "judger":

                user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}

Your response:
"""

        else:
            if role == "planner":
                user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}

Your response:
"""
    
            elif role == "critic":
                user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}     

Your response:
"""
    
            elif role == "refiner":
                user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}

Your response:       
"""
            elif role == "judger":

                user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}

Your response:
"""

    elif args.task in ["mbppplus", "humanevalplus"]:
        
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 
    
Input Question: {question}

Your response:
"""

    elif args.task in ["winogrande"]:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_sequential_text_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert method in ["text_mas"], "only for text_mas method"
    assert "qwen" in args.model_name.lower(), "only for qwen models"

    # truncate context if needed
    ctx = context[: args.text_mas_context_length]

    if role == "planner":
        user_content = f"""
You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

## Input Question:
{question}

Your outlined plan should be concise with a few bullet points for each step. Do not produce the final answer.

## Format your response as follows:
Planner Agent's Output:
[Your detailed plan here]

Now output your plan to solve the question below:
"""

    elif role == "critic":
        user_content = f"""
You are a Critic Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan in text format.

Your job is to carefully evaluate the correctness and completeness of the plan and provide helpful feedback.

## Input Question:
{question}

## Plan from Planner Agent:
{ctx}

## Format your response as follows:
Critic Agent's Output:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""

    elif role == "refiner":
        user_content = f"""
You are a Refiner Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan together with Critic Agent's feedback in text format.

Your job is to incorporate the feedback and produce an improved, refined step-by-step plan.

## Input Question:
{question}

## Original Plan and Critic Feedback:
{ctx}

## Format your response as follows:
Refiner Agent's Output:
[Your refined and improved plan here]

Make sure your output plan is logically correct, concise, and sufficient to guide final problem solving.
Now, output your refined plan below:
"""

    elif role == "judger":
        task = getattr(args, "task", None)

        if task in ["gsm8k", "aime2024", "aime2025"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif task in ["mbppplus", "humanevalplus"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
"""
            
        elif task in ["winogrande"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""
        else:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and present your final answer clearly at the end.
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_hierarchical_text_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    
    assert method in ["text_mas"], "this prompt only for text_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"
    
    if args.task in ['gsm8k', 'aime2024', 'aime2025']:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

Input Question: {question}

Your response:
"""

    elif args.task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
        elif role == "judger":

            user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

Input Question: {question}

Your response:
"""

    elif args.task in ["mbppplus", "humanevalplus"]:
        
        if role == "planner":
            user_content = f"""
You are a math agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "critic":
            user_content = f"""
You are a science agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "refiner":
            user_content = f"""
You are a code agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the final answer in markdown python code block.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

Input Question: {question}

Your response:
"""

    elif args.task in ["winogrande"]:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_single_agent(question: str, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert args.method in ["baseline"], "this prompt only for baseline method (single agent)"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    task = args.task

    if task in ["gsm8k", "aime2024", "aime2025"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["mbppplus", "humanevalplus"]:
        user_content = f"""
Target Question: {question}

You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
Now, reason step by step and output the final answer:
"""

    elif task in ["winogrande"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    else:
        user_content = f"""
Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the question without outputting other irrelevant information.
Present your reasoning, and then clearly state your final answer at the end.
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


# ============= Document Extraction Dataset Prompts =============

# DocRED relation types (Wikidata property IDs) - Compact version
DOCRED_RELATIONS_COMPACT = """Key relation types (Wikidata IDs):
Geographic: P17(country), P131(located in), P150(contains), P36(capital), P1376(capital of), P276(location)
People: P19(birthplace), P20(death place), P27(citizenship), P569(birth date), P570(death date), P172(ethnic group)
Organization: P159(headquarters), P112(founded by), P127(owned by), P137(operator), P488(chairperson)
Family: P22(father), P25(mother), P26(spouse), P40(child), P3373(sibling)
Position: P6(head of gov), P35(head of state), P39(position held), P108(employer)
Creative: P50(author), P57(director), P58(screenwriter), P86(composer), P170(creator), P175(performer)
Language: P37(official language), P1412(languages spoken)
Other: P31(instance of), P279(subclass of), P361(part of), P527(has part)
Full list: P6,P17,P19,P20,P22,P25,P26,P27,P30,P31,P35,P36,P37,P39,P40,P50,P54,P57,P58,P69,P86,P102,P108,P112,P118,P123,P127,P131,P136,P137,P140,P150,P155,P156,P159,P161,P162,P166,P170,P171,P172,P175,P176,P178,P179,P190,P194,P205,P206,P241,P264,P272,P276,P279,P355,P361,P364,P400,P403,P449,P463,P488,P495,P527,P551,P569,P570,P571,P576,P577,P580,P582,P585,P607,P674,P676,P706,P710,P737,P740,P749,P800,P807,P840,P937,P1001,P1056,P1198,P1336,P1344,P1365,P1366,P1376,P1412,P1441,P3373"""

def build_extraction_prompts_sequential(dataset: str, role: str, question: str, item: dict, method=None, args=None):
    """
    Unified extraction prompts for DocRED/CORD/FUNSD/FinER datasets (Sequential mode).
    """
    import json
    
    system_message = "You are Qwen, created by Alibaba Cloud. You are a document information extraction specialist. When asked to extract, output ONLY valid JSON without any thinking process or explanatory text."
    
    assert method in ["latent_mas"], "this prompt only for latent_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"
    
    template = json.loads(item.get("extract_template", "{}"))
    template_str = json.dumps(template, ensure_ascii=False, indent=2)
    chunk_info = item.get("chunk_info", "")
    
    # Dataset-specific instructions
    if dataset == "docred":
        task_desc = "document-level relation extraction using Wikidata property IDs."
        focus_areas = "named entities (persons, organizations, locations, dates, etc.) and their relationships"
        output_constraint = f"""CRITICAL: Choose the CORRECT relation ID for each relationship:

Organization relations:
- P749: parent_organization (Company A → Company B means A is part of B)
- P355: subsidiary (Company B → Company A means A is subsidiary of B)
- P361: part_of (Department → Company)
- P108: employer (Person → Company where they work)
- P112: founder (Person who founded → Organization)

Location relations:
- P17: country (City/Organization → Country)
- P131: located_in (City/Building → Administrative area)
- P159: headquarters (Organization → City where HQ is located)
- P30: continent (Country/Region → Continent)

Time relations:
- P571: inception/founded (Organization → Date when established)
- P569: birth_date (Person → Birth date)
- P580: start_time (Event → Start date)

Examples with correct IDs:
{{"head": "Apple Inc.", "relation": "P749", "tail": "Apple Computer Company", "evidence": [0]}}
{{"head": "Steve Jobs", "relation": "P108", "tail": "Apple Inc.", "evidence": [1]}}
{{"head": "Cupertino", "relation": "P17", "tail": "United States", "evidence": [0]}}
{{"head": "IBM Research Brazil", "relation": "P571", "tail": "June 2010", "evidence": [2]}}

All valid IDs: P6,P17,P19,P20,P22,P25,P26,P27,P30,P31,P35,P36,P37,P39,P40,P50,P54,P57,P58,P69,P86,P102,P108,P112,P118,P123,P127,P131,P136,P137,P140,P150,P155,P156,P159,P161,P162,P166,P170,P171,P172,P175,P176,P178,P179,P190,P194,P205,P206,P241,P264,P272,P276,P279,P355,P361,P364,P400,P403,P449,P463,P488,P495,P527,P551,P569,P570,P571,P576,P577,P580,P582,P585,P607,P674,P676,P706,P710,P737,P740,P749,P800,P807,P840,P937,P1001,P1056,P1198,P1336,P1344,P1365,P1366,P1376,P1412,P1441,P3373"""
    
    elif dataset == "cord":
        task_desc = "receipt/invoice information extraction. Extract structured purchase information."
        focus_areas = "menu items (names, prices, counts), subtotal, total amount, tax, payment details"
        output_constraint = "Fill the 'menu' array with all items, and populate subtotal/total/tax fields with exact numeric values."
    
    elif dataset == "funsd":
        task_desc = "form understanding and key-value extraction. Extract form fields and their relationships."
        focus_areas = "form entities (questions, answers, headers, other text), spatial relationships between fields"
        output_constraint = "Fill 'entities' array with all text elements and their labels, 'relations' array with question-answer pairs."
    
    elif dataset == "finer":
        task_desc = "fine-grained financial entity recognition. Identify and classify financial domain entities."
        focus_areas = "financial terms (stock codes, rates, accounting items, metrics, dates, amounts, company names)"
        output_constraint = "Fill 'entities' array with all financial entities, their types, and positions in text."
    
    else:
        task_desc = "document information extraction."
        focus_areas = "key information elements in the document"
        output_constraint = "Fill all schema fields with extracted information."
    
    if role == "planner":
        user_prompt = f"""You are a Document Scanner Agent (Phase 1: Information Discovery).

Task: {task_desc}

Document Section {chunk_info}:
{question}

Instructions:
- Carefully read the document section
- Identify ALL relevant information: {focus_areas}
- Note context and relationships between information elements
- Store findings in latent format (do NOT output final JSON yet)
- Be thorough and precise

Begin scanning:
"""
    
    elif role == "critic":
        user_prompt = f"""You are a Document Validator Agent (Phase 2: Cross-Verification).

Task: {task_desc}

You have latent information from the previous scanning phase.

Document Section {chunk_info}:
{question}

Instructions:
- Cross-check extracted information against document content
- Verify accuracy of: {focus_areas}
- Identify any missing or ambiguous information
- Resolve inconsistencies
- Refine latent representation (do NOT output final JSON yet)

Continue verification:
"""
    
    elif role == "refiner":
        user_prompt = f"""You are a Document Structuring Agent (Phase 3: Organization).

Task: {task_desc}

You have latent information from previous agents.

Document Section {chunk_info}:
{question}

Instructions:
- Organize all extracted information logically
- Resolve any conflicts between different document sections
- Prepare comprehensive latent summary
- Focus on: {focus_areas}
- Refine latent representation (do NOT output final JSON yet)

Continue organization:
"""
    
    elif role == "judger":
        # Get entity list for DocRED
        entity_list = item.get("entity_list", "")
        docred_entity_section = ""
        if dataset == "docred" and entity_list:
            docred_entity_section = f"""
Entities in document (use these exact names):
{entity_list}
"""
        
        user_prompt = f"""You are the Final Extraction Agent.

IMPORTANT: Output ONLY the JSON object. Do NOT output any thinking process, explanations, or text outside the JSON.

Task: {task_desc}

{docred_entity_section}Document:
{question}

Schema:
{template_str}

{output_constraint}

Output rules:
1. Choose the CORRECT relation ID based on the relationship type (see categories above)
2. Use EXACT entity names from entity list - no typos, no spaces added/removed
3. Format: {{"head": "Entity Name", "relation": "P123", "tail": "Entity Name", "evidence": [0, 1]}}
4. evidence: sentence numbers starting from 0
5. Output ONLY valid JSON - no thinking process, no extra text

Extract:
"""
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]


def build_extraction_prompts_hierarchical(dataset: str, role: str, question: str, item: dict, method=None, args=None):
    """
    Unified extraction prompts for DocRED/CORD/FUNSD/FinER datasets (Hierarchical mode).
    """
    import json
    
    system_message = "You are Qwen, created by Alibaba Cloud. You are a document information extraction specialist. When asked to extract, output ONLY valid JSON without any thinking process or explanatory text."
    
    assert method in ["latent_mas"], "this prompt only for latent_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"
    
    template = json.loads(item.get("extract_template", "{}"))
    template_str = json.dumps(template, ensure_ascii=False, indent=2)
    partition_info = item.get("partition_info", "")
    
    # Dataset-specific instructions
    if dataset == "docred":
        task_desc = "document-level relation extraction using Wikidata property IDs"
        focus_areas = "named entities and their relationships using Wikidata IDs"
        docred_hint = f"\n\n{DOCRED_RELATIONS_COMPACT}"
    
    elif dataset == "cord":
        task_desc = "receipt/invoice extraction"
        focus_areas = "menu items, prices, totals, tax"
    
    elif dataset == "funsd":
        task_desc = "form understanding"
        focus_areas = "form fields, questions, answers, relationships"
    
    elif dataset == "finer":
        task_desc = "financial entity recognition"
        focus_areas = "financial domain entities and terms"
    
    else:
        task_desc = "document extraction"
        focus_areas = "key information"
    
    if role == "planner":
        user_prompt = f"""You are Document Partition Reader 1.

Task: {task_desc}

Your assigned: {partition_info}

Document Content:
{question}

Instructions:
- Extract ALL {focus_areas} from this partition only
- Be thorough and accurate
- Store in latent format (no final output yet)

Partition 1 extraction:
"""
    
    elif role == "critic":
        user_prompt = f"""You are Document Partition Reader 2.

Task: {task_desc}

Your assigned: {partition_info}

Document Content:
{question}

Instructions:
- Extract ALL {focus_areas} from this partition only
- Be thorough and accurate
- Store in latent format (no final output yet)

Partition 2 extraction:
"""
    
    elif role == "refiner":
        user_prompt = f"""You are Document Partition Reader 3.

Task: {task_desc}

Your assigned: {partition_info}

Document Content:
{question}

Instructions:
- Extract ALL {focus_areas} from this partition only
- Be thorough and accurate
- Store in latent format (no final output yet)

Partition 3 extraction:
"""
    
    elif role == "judger":
        # Build DocRED-specific instructions
        docred_instructions = ""
        # Get entity list for DocRED
        entity_list = item.get("entity_list", "")
        docred_instructions = ""
        if dataset == "docred":
            entity_section = ""
            if entity_list:
                entity_section = f"""
Named Entities (use EXACT names from this list):
{entity_list}
"""
            docred_instructions = f"""
CRITICAL: Choose the CORRECT relation ID for each relationship:

Organization: P749:parent_org, P355:subsidiary, P361:part_of, P108:employer, P112:founder
Location: P17:country, P131:located_in, P159:headquarters, P30:continent
Time: P571:inception, P569:birth_date, P580:start_time

Examples:
{{"head": "Apple Inc.", "relation": "P749", "tail": "Apple Computer", "evidence": [0]}}
{{"head": "Steve Jobs", "relation": "P108", "tail": "Apple Inc.", "evidence": [1]}}
{{"head": "Cupertino", "relation": "P17", "tail": "United States", "evidence": [0]}}

{entity_section}
Output rules:
1. Choose CORRECT relation ID (see categories above)
2. Use EXACT entity names - no typos
3. Format: {{"head": "Name", "relation": "P123", "tail": "Name", "evidence": [0]}}
4. evidence: sentence numbers from 0
5. No thinking process, output JSON only
"""
        
        user_prompt = f"""You are the Final Synthesizer.

IMPORTANT: Output ONLY the JSON object. Do NOT output any thinking process, explanations, or text outside the JSON.

Task: {task_desc}

Full Document Content:
{question}

You have latent information from all document partitions analyzing this content.

Target Extraction Schema:
{template_str}
{docred_instructions}
Instructions:
1. Merge information from all partitions - choose CORRECT relation IDs
2. Remove duplicate entries
3. Use EXACT entity names - no typos
4. Format: {{"head": "Name", "relation": "P123", "tail": "Name", "evidence": [0]}}
5. Output ONLY valid JSON - no thinking, no extra text

Final extraction (JSON only):
"""
    
    # Check if item has image (multimodal)
    if "image" in item and item["image"] is not None:
        # Multimodal format: image + text
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [
                {"type": "image", "image": item["image"]},  # PIL Image object
                {"type": "text", "text": user_prompt}
            ]},
        ]
    else:
        # Text-only format
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]


def build_multimodal_extraction_message(role: str, image, text_prompt: str, system_message: str = None):
    """
    Helper function to build multimodal messages with image and text.
    Used for vision-language models like Qwen-VL.
    
    Args:
        role: Agent role (planner/critic/refiner/judger)
        image: PIL Image object
        text_prompt: Text instruction
        system_message: System prompt (optional)
    
    Returns:
        List of message dicts in Qwen-VL format
    """
    if system_message is None:
        system_message = "You are Qwen, created by Alibaba Cloud. You are a document information extraction specialist."
    
    messages = [{"role": "system", "content": system_message}]
    
    if image is not None:
        # Multimodal: image + text
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt}
            ]
        })
    else:
        # Text-only fallback
        messages.append({
            "role": "user",
            "content": text_prompt
        })
    
    return messages


def build_extraction_prompts_text_mas_sequential(dataset: str, role: str, question: str, context: str, item: dict, method=None, args=None):
    """
    Text-MAS extraction prompts for document extraction (Sequential mode).
    Uses explicit text passing between agents.
    """
    import json
    
    system_message = "You are Qwen, created by Alibaba Cloud. You are a document information extraction specialist. When asked to extract, output ONLY valid JSON without any thinking process or explanatory text."
    
    assert method in ["text_mas"], "this prompt only for text_mas method"
    
    template = json.loads(item.get("extract_template", "{}"))
    template_str = json.dumps(template, ensure_ascii=False, indent=2)
    chunk_info = item.get("chunk_info", "")
    
    # Dataset-specific instructions
    if dataset == "docred":
        task_desc = "document-level relation extraction using Wikidata property IDs"
        focus_areas = "named entities and their relationships (using Wikidata relation IDs like P17, P569, P27)"
        output_constraint = f"""CRITICAL: The 'relation' field MUST be a Wikidata property ID.
The 'evidence' field is a list of sentence indices (0-indexed) that support the relation.
{DOCRED_RELATIONS_COMPACT}

Examples:
- {{"head": "Barack Obama", "relation": "P19", "tail": "Honolulu", "evidence": [0, 1]}} (place of birth)
- {{"head": "Paris", "relation": "P17", "tail": "France", "evidence": [2]}} (country)"""
    elif dataset == "cord":
        task_desc = "receipt/invoice information extraction"
        focus_areas = "menu items, prices, subtotal, total, tax"
        output_constraint = "Fill menu array with items, populate subtotal/total/tax fields."
    elif dataset == "funsd":
        task_desc = "form understanding and key-value extraction"
        focus_areas = "form fields (questions, answers, headers)"
        output_constraint = "Fill entities and relations arrays."
    elif dataset == "finer":
        task_desc = "financial named entity recognition"
        focus_areas = "financial terms, amounts, companies, metrics"
        output_constraint = "Fill entities array with all financial entities."
    else:
        task_desc = "document information extraction"
        focus_areas = "key information"
        output_constraint = "Fill all schema fields."
    
    if role == "planner":
        user_prompt = f"""You are a Document Scanner Agent.

Task: {task_desc}

Document Section {chunk_info}:
{question}

Instructions:
1. Carefully scan the document
2. Identify all relevant information: {focus_areas}
3. List all found items with their values
4. Be thorough - don't miss anything

Output your findings in a structured list format.
"""
    
    elif role == "critic":
        user_prompt = f"""You are a Document Validator Agent.

Task: {task_desc}

Document Section {chunk_info}:
{question}

Previous Agent's Analysis:
{context}

Instructions:
1. Review the previous agent's findings
2. Cross-check against the document content
3. Identify any missing or incorrect information
4. Add corrections and missing items

Output your verified and corrected findings.
"""
    
    elif role == "refiner":
        user_prompt = f"""You are a Document Organizer Agent.

Task: {task_desc}

Document Section {chunk_info}:
{question}

Previous Agents' Analysis:
{context}

Instructions:
1. Organize all verified information
2. Resolve any conflicts between findings
3. Prepare comprehensive summary
4. Focus on: {focus_areas}

Output the organized, complete list of extracted information.
"""
    
    elif role == "judger":
        # Get entity list for DocRED
        entity_list = item.get("entity_list", "")
        docred_entity_section = ""
        if dataset == "docred" and entity_list:
            docred_entity_section = f"""
Named Entities (use EXACT names):
{entity_list}
"""
        
        user_prompt = f"""You are the Final Extraction Agent.

IMPORTANT: Output ONLY the JSON object. Do NOT output any thinking process, explanations, or text outside the JSON.

Task: {task_desc}
{docred_entity_section}
Document: {question}

Schema: {template_str}

{output_constraint}

Output rules:
1. Choose CORRECT relation ID based on relationship type
2. Use EXACT entity names - no typos or extra spaces
3. Format: {{"head": "Entity Name", "relation": "P123", "tail": "Entity Name", "evidence": [0, 1]}}
4. evidence: sentence numbers starting from 0
5. Output ONLY valid JSON - no thinking, no extra text

Previous agents' analysis:
{context}

Extract:
"""
    
    # Check if item has image (multimodal)
    if "image" in item and item["image"] is not None:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": user_prompt}
            ]},
        ]
    else:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]


def build_extraction_prompts_text_mas_hierarchical(dataset: str, role: str, question: str, context: str, item: dict, method=None, args=None):
    """
    Text-MAS extraction prompts for document extraction (Hierarchical mode).
    Uses explicit text passing between agents, with parallel partition processing.
    """
    import json
    
    system_message = "You are Qwen, created by Alibaba Cloud. You are a document information extraction specialist. When asked to extract, output ONLY valid JSON without any thinking process or explanatory text."
    
    assert method in ["text_mas"], "this prompt only for text_mas method"
    
    template = json.loads(item.get("extract_template", "{}"))
    template_str = json.dumps(template, ensure_ascii=False, indent=2)
    partition_info = item.get("partition_info", "")
    
    # Dataset-specific instructions (same as sequential)
    if dataset == "docred":
        task_desc = "document-level relation extraction using Wikidata property IDs"
        focus_areas = "named entities and their relationships (using Wikidata relation IDs like P17, P569, P27)"
        output_constraint = f"""CRITICAL: The 'relation' field MUST be a Wikidata property ID.
The 'evidence' field is a list of sentence indices (0-indexed) that support the relation.
{DOCRED_RELATIONS_COMPACT}

Examples:
- {{"head": "Barack Obama", "relation": "P19", "tail": "Honolulu", "evidence": [0, 1]}} (place of birth)
- {{"head": "Paris", "relation": "P17", "tail": "France", "evidence": [2]}} (country)"""
    elif dataset == "cord":
        task_desc = "receipt/invoice information extraction"
        focus_areas = "menu items, prices, subtotal, total, tax"
        output_constraint = "Fill menu array with items, populate subtotal/total/tax fields."
    elif dataset == "funsd":
        task_desc = "form understanding and key-value extraction"
        focus_areas = "form fields (questions, answers, headers)"
        output_constraint = "Fill entities and relations arrays."
    elif dataset == "finer":
        task_desc = "financial named entity recognition"
        focus_areas = "financial terms, amounts, companies, metrics"
        output_constraint = "Fill entities array with all financial entities."
    else:
        task_desc = "document information extraction"
        focus_areas = "key information"
        output_constraint = "Fill all schema fields."
    
    if role == "planner":
        user_prompt = f"""You are Document Partition Reader 1.

Task: {task_desc}

Your Partition: {partition_info}

Document Content:
{question}

Instructions:
1. Extract all relevant information from YOUR partition
2. Focus on: {focus_areas}
3. List findings with exact values
4. Note cross-references to other sections

Your partition findings:
"""
    
    elif role == "critic":
        user_prompt = f"""You are Document Partition Reader 2.

Task: {task_desc}

Your Partition: {partition_info}

Document Content:
{question}

Other Partition's Findings:
{context}

Instructions:
1. Extract information from YOUR partition
2. Cross-reference with other partition's findings
3. Add complementary information

Your partition findings:
"""
    
    elif role == "refiner":
        user_prompt = f"""You are Document Partition Reader 3.

Task: {task_desc}

Your Partition: {partition_info}

Document Content:
{question}

All Previous Findings:
{context}

Instructions:
1. Extract remaining information from YOUR partition
2. Merge with previous findings
3. Resolve any conflicts
4. Create unified summary

Merged findings:
"""
    
    elif role == "judger":
        # Get entity list for DocRED
        entity_list = item.get("entity_list", "")
        docred_entity_section = ""
        if dataset == "docred" and entity_list:
            docred_entity_section = f"""
Named Entities (use EXACT names):
{entity_list}
"""
        
        user_prompt = f"""You are the Final Extraction Agent.

IMPORTANT: Output ONLY the JSON object. Do NOT output any thinking process, explanations, or text outside the JSON.

Task: {task_desc}
{docred_entity_section}
Document: {question}

Schema: {template_str}

{output_constraint}

Output rules:
1. Merge partition findings - choose CORRECT relation ID for each
2. Use EXACT entity names - no typos
3. Format: {{"head": "Name", "relation": "P123", "tail": "Name", "evidence": [0]}}
4. Remove duplicates
5. Output ONLY valid JSON - no thinking, no extra text

Partition findings:
{context}

Extract:
"""
    
    # Check if item has image (multimodal)
    if "image" in item and item["image"] is not None:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": user_prompt}
            ]},
        ]
    else:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]
