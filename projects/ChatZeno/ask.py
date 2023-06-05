import openai
import pandas as pd
import numpy as np
import openai.embeddings_utils as eu
import os

system_2 = '''The user ask question about how to use the Zeno simulation and graphics node system. For each user question, answer what is good search keyword(s) for the question.

Q: How to simulate tsumani?
A: tsumani simulation

Q: Simulate bridge breakdown.
A: bridge break

Q: Please create a cube in the scene.
A: create cube

Q: Create a scene that fluid emits from a box collides with a cylinder below it
A: fluid box emits collides cylinder translate

Q: Could you make an animation that a cube move from left to right?
A: cube motion animation

Q: Make the dragon model flaming fires.
A: dragon model flame fire

Q: Load my mesh model at C:/Learning/testMonkey.obj
A: load mesh model

Q: Translate this primitive with offset (0, 0, 1)
A: translate primitive offset

Q: Please load the ocean volume from myfile.vdb
A: load ocean volume vdb

Q: {}
A:'''

system_5 = '''You are going to assist the user by helping them manipulate the Zeno node system. Each node acts like a function, it have several arguments as input, and only one output, represented by their node ID. As an AI language model, you could not operate the Zeno software (which requries GUI) directly. However, with the power of Zeno's Python API, you can control nodes and create links indirectly!

Zeno is a 3D computer graphics software with an Python API. It uses a right-handed XYZ coordinate system, where +Z for up, +Y for front and +X for right. Coordinates (aka 3D vectors) are represented in Python tuples with 3 floating number elements. Zeno is consist of nodes, nodes can be invoked from Python API, nodes are provided as Python functions, node arguments must be specified as keyword arguments, for example:
```python
import zeno
result = zeno.NodeName(arg1=val1, arg2=val2).out1
finalResult = zeno.AnotherNodeName(arg1=result).result
```

Your output should be Python scripts in this format:
```python
Your answer goes here.
```
Only answer the Python script, without any additional text, no nature languages.
'''

def extract_keyword(question):
    completion = openai.Completion.create(
        engine='text-davinci-003',
        prompt=system_2.format(question),
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )
    answer = completion.choices[0].text.strip()  # type: ignore
    return answer

class Conversation:
    def __init__(self):
        self.messages = [
            {'role': 'system', 'content': system_5},
        ]

    def ask(self, question, node_refs=[]):
        node_refs = list(node_refs)
        if node_refs:
            question += '\nYou may need these nodes:\n' + '\n'.join(node_refs)
        self.messages.append({'role': 'user', 'content': question})
        completion = openai.ChatCompletion.create(
            messages=self.messages,
            model='gpt-3.5-turbo',
            temperature=1,
        )
        answer = completion.choices[0].message.content  # type: ignore
        return answer

class APIReference:
    def __init__(self):
        node_names = []
        if os.path.exists('refs.json'):
            self.df = pd.read_json('refs.json', orient='columns')
            print('==> Loading refs.json')
            return
        for entry in os.listdir('refs'):
            if entry.endswith('.md') and entry.startswith('Prim'):
                node_names.append(entry[:-len('.md')])
        sections = []
        # node_names = ['PrimitiveResize', 'PrimitiveScale']
        for node in node_names:
            with open(os.path.join('refs', node + '.md'), 'r') as f:
                sections.append((node, f.read().strip()))
        self.df = pd.DataFrame(np.array(sections), columns=['title', 'content'])
        def do_embed(x):
            print(f'==> Processing {x[:50]}...')
            return eu.get_embedding(x, engine='text-embedding-ada-002')
        self.df["embedding"] = self.df.title.apply(do_embed)
        print('==> Saving refs.json')
        self.df.to_json('refs.json', orient='records')

    def search_keyword(self, keyword, n=3):
        key_embed = eu.get_embedding(keyword, engine='text-embedding-ada-002')
        self.df["similarity"] = self.df.embedding.apply(lambda x: eu.cosine_similarity(x, key_embed))
        results = self.df.sort_values("similarity", ascending=False).head(n).content
        return results

refs = APIReference()

# print(system)
# question = "How to translate a primitive with offset (0, 0, 1)?"
question = input('question? ')
nodes = []
kwd = extract_keyword(question)
print('==> Keywords are:', kwd)
results = refs.search_keyword(kwd, n=1)
print(results)
results = list(results)
if len(results) > 1:
    for i in range(len(results)):
        nodes.append(results[i])
        print('==> Thinking...')
        print(Conversation().ask(question, nodes))
        is_missing = input('missing? (y/N) ')
        if not is_missing:
            break
nodes = [results[0]]
print('==> Thinking...')
print(Conversation().ask(question, nodes))
for times in range(10):
    keys = input('missing? ').strip()
    if not keys:
        break
    keys = keys.split(' ')
    for key in keys:
        results = refs.search_keyword(key, n=1)
        print(results)
        nodes += list(results)
    print('==> Thinking...')
    print(Conversation().ask(question, nodes))
