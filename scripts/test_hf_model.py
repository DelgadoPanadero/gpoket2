from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

# Load model
ckpt = snapshot_download("iamthinbaker/GPokeT2", revision="v0.1-wip-4200")
tokenizer = PreTrainedTokenizerFast.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)


# Generate Pokemon!!!
image = model.generate_sprite(
    tokenizer, 
    type1="fire", 
    type2="dragon", 
    verbose=True,
)
cv2.imwrite("pokemon.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))