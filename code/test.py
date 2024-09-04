from factory import NewModule
from dotenv import load_dotenv

load_dotenv()

model_name = 'SAG'
model = NewModule(model_name)
for name, param in model.named_parameters():
    print(name)
