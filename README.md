# StableDiffusionInference
*To be able to run this previously you have to have your `.ckpt` model generated and fully functional with [Stable Diffusion](https://github.com/CompVis/stable-diffusion)*

*StableDiffusionInference* is a reduced way to reproduce our previously saved models in [Hugging Face](https://huggingface.co/) without working on their online or local tool ([NMKD Stable Diffusion GUI](https://nmkd.itch.io/t2i-gui)). In this way, we work locally thanks to download SD using [diffusers library](https://github.com/huggingface/diffusers/tree/main#new--stable-diffusion-is-now-fully-compatible-with-diffusers) without Google Drive or having any available space. 

In this example, I will use my dog *Urko* to show all the process:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A-ezqHDd37q3ga6Aqqn_mw9xFngAlDz4#scrollTo=1p7rlojPgOFW)

## Requirements
You need it to update, install and run *Hugging Face*

```python
pip install diffusers==0.7.2
transformers==4.24.0
huggingface_hub==0.10.1
```

## Hugging Face

1. Register in [Hugging Face](https://huggingface.co/) and create a [token](https://huggingface.co/settings/tokens) with write access. 
For more information: https://huggingface.co/docs/hub/security-tokens

2. Create a Model in your profile using [Diffusers](https://huggingface.co/spaces/diffusers/convert-sd-ckpt). Convert your Stable Diffusion `.ckpt` file to Hugging Face Diffusers (Select *Host the model on the Hugging Face Hub* in 2nd step)

3. Upload your `.ckpt` in your Hugging Face model:

![image](https://user-images.githubusercontent.com/23172965/204022430-31714cd5-ca1e-4e2c-adc3-a0d34988cc2c.png)

4. Use your model name for your `model_id` later:

![image](https://user-images.githubusercontent.com/23172965/204022618-23b26ee1-4075-422b-a230-19dc71896e9d.png)

## Configuration

```python
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 50
```

## Text2image 

### Individual image 🏞
```python
model_id = "MAVS/urkov1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "urko_dog in the style of 90s anime, bright,red flowers, foot path, trees, award winning, trending on artstation"

image = pipe(prompt, height=HEIGHT, width=WIDTH, num_inference_steps=NUM_INFERENCE_STEPS).images[0]
image
```

### Multiple images in figure mode 🌆🌆🌆
```python
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


## RESULTS

● Real images of Urko:

![image](https://user-images.githubusercontent.com/23172965/204024695-f82daf7b-bb89-4cc5-8c67-7cd36f27a2cd.png)
![image](https://user-images.githubusercontent.com/23172965/204024777-8a100bae-7c55-430f-b51c-deadb7051705.png)
![image](https://user-images.githubusercontent.com/23172965/204024843-589d4a6d-d2ab-41f6-8025-bfbfde43c36d.png)

● Individual images of IA:

![image](https://user-images.githubusercontent.com/23172965/204024470-e7c693f5-5c17-4b06-8aeb-67d010dfa095.png)
![image](https://user-images.githubusercontent.com/23172965/204024491-4ac4a02f-8d9b-48f5-8eff-372309e1b6e3.png)
![image](https://user-images.githubusercontent.com/23172965/204024565-da89afa8-6e96-4187-9aca-75d9adc9f12c.png)

● Images in figure mode

![image](https://user-images.githubusercontent.com/23172965/204028172-0a98856b-2b13-41bc-808e-7ea7c7176d45.png)
