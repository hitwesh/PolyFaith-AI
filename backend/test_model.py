import time

from backend.models.model_loader import ModelLoader

print("Loading model...")

model, tokenizer = ModelLoader.load()

print("Model ready.")

prompt = "Answer briefly: What is free will?"

messages = [
    {
        "role": "user",
        "content": prompt
    }
]

print("Creating prompt...")

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("Tokenizing...")

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)

print("Generating response...")

start = time.time()

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

elapsed = time.time() - start

print(f"Generation took {elapsed:.2f} seconds")

response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)

print(tokenizer.decode(outputs[0]))

print("\n" + "=" * 50)
print(response)
print("=" * 50)