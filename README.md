# SD-XL: Generador de imágenes con Stable Diffusion XL

Proyecto en Python que genera imágenes a partir de texto usando el modelo [Stable Diffusion XL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) de Stability AI. Funciona con GPU NVIDIA (CUDA) y, si no hay una disponible, usa la CPU.

## Ejemplo

![Imagen generada con SD-XL](imagen.png)

**Prompt:** `escribe aquí el prompt que usaste`

## Tecnologías

- **Python 3.12**
- **uv**: gestión de dependencias y entorno virtual
- **PyTorch** (CUDA 13.0): ejecución del modelo en GPU
- **Diffusers**: pipeline de Stable Diffusion XL
- **Transformers**: codificadores de texto (CLIP)
- **Accelerate**: carga del modelo por partes entre GPU y RAM
- **Safetensors**: carga segura de los pesos del modelo

## Requisitos

- Windows con GPU NVIDIA y drivers actualizados (probado en RTX 3050 Laptop de 4 GB)
- [uv](https://docs.astral.sh/uv/) instalado
- Unos 10 GB libres en disco para el modelo
- Mínimo 16 GB de RAM si se ejecuta en CPU

## Instalación

1. Clonar el repositorio:

```bash
   git clone <url-del-repo>
   cd SD-XL
```

2. Instalar las dependencias:

```bash
   uv sync
```

3. Verificar que PyTorch detecta la GPU:

```bash
   uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

   Si sale `True`, el modelo se ejecutará en la GPU.

## Uso

```bash
uv run sd-xl
```

El programa pide un prompt por consola, genera la imagen y la guarda como `imagen.png` en la carpeta del proyecto.

La primera ejecución descarga el modelo desde Hugging Face (unos 7 GB), así que se demora un poco más.

## Cómo funciona

- **Detección del dispositivo:** si hay GPU NVIDIA usa CUDA con precisión `float16`; si no, usa la CPU con `float32`.
- **VAE corregido:** se usa `madebyollin/sdxl-vae-fp16-fix`, porque el VAE original de SDXL puede generar imágenes negras en `float16`.
- **Ahorro de memoria:** en GPU se activa `enable_model_cpu_offload()`, que permite correr SDXL con solo 4 GB de VRAM. En CPU se usa `enable_attention_slicing()` para reducir el consumo de RAM.
- **Parámetros de generación:** 25 pasos de inferencia, `guidance_scale` de 7.0 y resolución de 1024x1024.

## Estructura del proyecto

```
SD-XL/
├── src/
│   └── sd_xl/
│       └── __init__.py   # Lógica de carga del modelo y generación
├── pyproject.toml        # Dependencias y configuración de uv
├── uv.lock
├── imagen.png            # Imagen generada de ejemplo
└── README.md
```

## Autor

**Darío Rivera**