from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_path = "./outmodel"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

def chat(user_input, max_length=80):
    prompt = f"User: {user_input}\nBot:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        temperature=0.5
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Bot:" in text:
        bot_part = text.split("Bot:")[-1]        
        if "User:" in bot_part:
            bot_part = bot_part.split("User:")[0]
        bot_reply = bot_part.strip()
    else:
        bot_reply = text.strip()

    return bot_reply

print("GPT2 對話測試（輸入 exit 離開）")
while True:
    msg = input("你：")
    if msg.strip().lower() == "exit":
        break
    print("GPT2：", chat(msg))
