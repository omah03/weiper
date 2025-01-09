from openood.networks import ResNet18_32x32
import torch
from openood.evaluation_api import Evaluator
from openood.preprocessors import BasePreprocessor
from tqdm import tqdm

from openood.networks import ResNet50
from torchvision.models import ResNet50_Weights
from torch.hub import load_state_dict_from_url
dataset = "cifar10"
models = [ResNet18_32x32(num_classes=100 if dataset=="cifar100" else 10)
          for _ in range(3)]
[model.load_state_dict(
    torch.load(f'./OpenOOD/results/{dataset}_resnet18_32x32_base_e100_lr0.1_default/s{i}/best.ckpt')
) for i,model in enumerate(models)]
from functools import wraps
def batch_processing_decorator(batch_size):
    def decorator(forward_fn):
        @wraps(forward_fn)
        def wrapped_forward(x, *args, **kwargs):
            # Split the input tensor into smaller batches
            batched_input = torch.split(x, batch_size, dim=0)
            
            # Initialize an empty list to store outputs for each batch
            outputs = []
            features = []
            has_feature = False
            # Process each batch independently
            for batch in tqdm(batched_input, disable=True):
                output = forward_fn(batch, *args, **kwargs)  # Call the original forward method
                if isinstance(output, tuple) and len(output) == 2:
                    outputs.append(output[0])
                    features.append(output[1])
                    has_feature = True
                else:
                    outputs.append(output)
                
            # Concatenate all batch outputs
            final_output = torch.cat(outputs, dim=0)
            if has_feature:
                final_features = torch.cat(features, dim=0)
                return final_output, final_features
            return final_output
        return wrapped_forward
    return decorator
        
    # Concatenate all batch outputs
    final_output = torch.cat(outputs, dim=0)
    return final_output
    

if dataset == 'imagenet':
    net = ResNet50()
    weights = ResNet50_Weights.IMAGENET1K_V1
    net.load_state_dict(load_state_dict_from_url(weights.url))
    
    net.forward =  batch_processing_decorator(batch_size=200)(net.forward)
    models = [net]
else:
    for m in models:
        m.forward = batch_processing_decorator(batch_size=200)(m.forward)
metrics = None
iterations = 1
for i in tqdm(range(iterations), disable=True):
    for j, model in enumerate(models):
        model.cuda()
        model.eval()

        evaluator = Evaluator(
            model, 
            id_name=dataset,
            data_root="./data",
            config_root="./OpenOOD/configs/",
            preprocessor=None, 
            postprocessor_name='weiper_kldiv', 
            batch_size=int(1e4),
            verbose=True, 
            APS_mode=False,
            num_workers=0,
        )
        evaluator.postprocessor.n_repeats = (
            100 if dataset == "cifar10" else 100
        )
        evaluator.eval_ood()
        if i == 0 and j == 0:
            metrics = evaluator.metrics["ood"]
            print(evaluator.postprocessor.get_hyperparam())
        else:
            metrics += evaluator.metrics["ood"] 
metrics/(len(models)*iterations)