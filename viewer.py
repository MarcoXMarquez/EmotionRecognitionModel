import torch
print("CUDA disponible:", torch.cuda.is_available())
print("Nombre GPU:", torch.cuda.get_device_name(0))
print("Memoria disponible:", torch.cuda.mem_get_info())