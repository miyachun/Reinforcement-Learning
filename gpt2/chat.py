from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

model_path = "./outmodel"  # 你剛剛訓練好的模型
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

def chat(user_input):
    prompt = f"User: {user_input}\nBot:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=80,         # top-k & top-p 控制創造力
        top_p=0.95,
        temperature=0.8,  # 溫度越高越有創意，但可能飄
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = text.split("Bot:")[-1].strip()
    return reply

print("GPT2 對話測試（輸入 exit 離開）")
while True:
    msg = input("你：")
    if msg.strip().lower() == "exit":
        break
    print("GPT2：", chat(msg))
