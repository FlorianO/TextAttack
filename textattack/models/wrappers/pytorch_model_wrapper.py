import torch
from torch.nn import CrossEntropyLoss
import textattack

from .model_wrapper import ModelWrapper


class PyTorchModelWrapper(ModelWrapper):
    """Loads a PyTorch model (`nn.Module`) and tokenizer."""

    def __init__(self, model, tokenizer, loss=CrossEntropyLoss(), batch_size=32):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"PyTorch model must be torch.nn.Module, got type {type(model)}"
            )

        self.model = model.to(textattack.shared.utils.device)
        self.tokenizer = tokenizer
        self.loss_fn = loss
        self.batch_size = batch_size

    def __call__(self, text_input_list):
        model_device = next(self.model.parameters()).device
        ids = self.tokenize(text_input_list)
        ids = torch.tensor(ids).to(model_device)

        with torch.no_grad():
            outputs = textattack.shared.utils.batch_model_predict(
                self.model, ids, batch_size=self.batch_size
            )

        return outputs

    def get_grads(self, text_input_list):
        """
        Get gradient w.r.t. embedding layer
        Args:
            text_input_list (list[str]): list of input strings
        Returns:
            dictionary
        """

        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must have method `get_input_embeddings` that returns `torch.nn.Embedding` object that represents input embedding layer"
            )

        self.model.train()

        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []
        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])
        
        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        ids = self.tokenize(text_input_list)
        
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            tokens = [self.tokenizer.convert_ids_to_tokens(_ids) for _ids in ids]
        else:
            tokens = ids
        
        ids = torch.tensor(ids).to(model_device)
        predictions = self.model(ids)
        
        original_label = predictions.argmax(dim=1)
        loss = self.loss_fn(predictions, original_label)
        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0]

        if len(grad.shape) > 2:
            grad = list(torch.transpose(grad, 0, 1).unbind(dim=0))
        else:
            grad = [grad]

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()

        output = {"tokens": tokens, "ids": ids.cpu(), "gradient": grad}

        return output
