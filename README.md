# StableDiffusionInference
StableDiffusionInference is a reduced way to reproduce our previously saved models in hugging Face without having to call the stablediffusion interface. In this way we work locally without having to enter Google Drive or having available space

-----

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A-ezqHDd37q3ga6Aqqn_mw9xFngAlDz4#scrollTo=1p7rlojPgOFW)

## Requirements
You need update and install to run HuggingFace

```
pip install diffusers==0.7.2
transformers==4.24.0
huggingface_hub==0.10.1
```

## Hugging Face

1. Register in (https://huggingface.co/) and create a token with write access (https://huggingface.co/settings/tokens). For more information: https://huggingface.co/docs/hub/security-tokens

2. Create a Model and transform your .CKPT to HuggingFace - Model

## CONFIGURATION

```
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 50
```

## Text2image 

### Individual image
```
model_id = "MAVS/urkov1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "urko_dog in the style of 90s anime, bright,red flowers, foot path, trees, award winning, trending on artstation"

image = pipe(prompt, height=HEIGHT, width=WIDTH, num_inference_steps=NUM_INFERENCE_STEPS).images[0]
image
```

### Multiple images in figure mode
```
prompt = "4K HD, high detail photograph, shot with Sigma f/ 4.2 , 250 mm sharp lens, shallow depth of field, subject= urko_dog sitting on the grass, consistent, high detailed light refraction, high level texture render"

rows, cols = 3, 3
fig = plt.figure (figsize=(8,8))

for i in range(0, rows*cols):
  image = pipe(prompt, height=HEIGHT, width=WIDTH, num_inference_steps=NUM_INFERENCE_STEPS).images[0]
  fig.add_subplot(rows,cols, i+1)
  plt.imshow(image)
  plt.axis('off')  

fig.tight_layout() #REMOVE WHITE PADDING
plt.savefig('pict.png', bbox_inches='tight', pad_inches = 0)
plt.show()
```
