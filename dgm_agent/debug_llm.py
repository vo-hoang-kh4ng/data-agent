import json
import sys
sys.path.append('d:\\kh4ng\\Data_agent')
from dotenv import load_dotenv
load_dotenv()
from dgm_agent.llm import create_client

with open('d:\\kh4ng\\Data_agent\\dgm_agent\\lcb_tasks\\tasks_100_val.json', encoding='utf-8') as f:
    tasks = json.load(f)
task = next((t for t in tasks if t['task_id'] == '2884'), None)

prompt = "You are an expert Data & Algorithm Agent.\nSolve the following programming problem.\n\n<problem_description>\n" + task['problem_description'] + "\n</problem_description>\n"
system_msg = 'You are an expert competitive programmer. Write clean, correct Python code that solves the given problem. Please wrap your code in a ```python ... ``` block.'

client, model = create_client('hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8')
resp = client.chat.completions.create(
    model=model,
    messages=[
        {'role': 'system', 'content': system_msg},
        {'role': 'user', 'content': prompt}
    ],
    temperature=0.7,
    max_tokens=4096,
    n=1
)
print('Finish reason:', resp.choices[0].finish_reason)
print('Content:', repr(resp.choices[0].message.content))
