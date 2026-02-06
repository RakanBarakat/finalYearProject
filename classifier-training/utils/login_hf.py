from huggingface_hub import login

print("Please enter your Hugging Face token.")
print("You can get one from: https://huggingface.co/settings/tokens")
print()

token = input("Enter your token: ").strip()
login(token=token)

print("\n✓ Successfully logged in to Hugging Face!")
