1. Dump descriptors:
```bash
build/bin/zenoedit -invoke dumpdesc | grep "^\[.*\]$" > desc.jsonl
```

2. Convert descriptors to docs:
```bash
python conv.py desc.jsonl -o refs
```

2. Ask GPT questions:
```bash
python ask.py "How to write a fluid simulation?"
```
