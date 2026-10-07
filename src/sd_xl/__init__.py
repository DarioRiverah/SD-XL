import torch  # Ejecutar operaciones matematicas del modelo de IA
from diffusers import AutoPipelineForText2Image  # Libreria especializada en modelos generativos


def main() -> None:
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    precision = torch.float16 if dispositivo == "cuda" else torch.float32
    print(f"Usando: {dispositivo}")

    print("Cargando el modelo ...")

    modelo = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=precision,  # fp16 en GPU, fp32 en CPU
        variant="fp16",         # descarga pesos fp16 (pesan la mitad)
        use_safetensors=True,
    )

    if dispositivo == "cuda":
        modelo.enable_model_cpu_offload()  # necesario con poca VRAM (4 GB)
    else:
        modelo = modelo.to(dispositivo)
        modelo.enable_attention_slicing()  # ahorra RAM en CPU

    prompt = input("Escribe el prompt de la imagen que quieres crear: ")
    #negative_prompt = "blurry, low quality, distorted, deformed, bad anatomy, watermark, text"

    print("Generando imagen ...")

    imagen = modelo(
        prompt=prompt,
        #negative_prompt=negative_prompt,
        num_inference_steps=25,
        guidance_scale=7.0,
        height=1024,
        width=1024,
    ).images[0]

    imagen.save("imagen.png")
    print("Imagen guardada exitosamente!")