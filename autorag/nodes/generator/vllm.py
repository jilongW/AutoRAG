import inspect
from copy import deepcopy
from typing import List, Tuple
import gc
import torch


from autorag.nodes.generator.base import generator_node


@generator_node
def vllm(prompts: List[str], llm: str, **kwargs) -> Tuple[List[str], List[List[int]], List[List[float]]]:
    """
    Vllm module.
    It gets the VLLM instance, and returns generated texts by the input prompt.
    You can set logprobs to get the log probs of the generated text.
    Default logprobs is 1.

    :param prompts: A list of prompts.
    :param llm: Model name of vLLM.
    :param kwargs: The extra parameters for generating the text.
    :return: A tuple of three elements.
        The first element is a list of generated text.
        The second element is a list of generated text's token ids.
        The third element is a list of generated text's log probs.
    """
    #print(1)
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput
    from vllm.sequence import SampleLogprobs
    input_kwargs = deepcopy(kwargs)
    #print(llm)
    vllm_model = LLM(model=llm, trust_remote_code=True)
    #print(11)
    if 'logprobs' not in input_kwargs:
        input_kwargs['logprobs'] = 1
    #print(input_kwargs)
    generate_params = SamplingParams(temperature=input_kwargs['temperature'],logprobs=input_kwargs['logprobs'], truncate_prompt_tokens=2047, min_tokens=1) 
    #print(111)
    #print(generate_params)
    #prompts = prompts[:10]
    #print("*************************")
    #print(len(prompts))
    results: List[RequestOutput] = vllm_model.generate(prompts, generate_params)
    #print("=============")
    #print(len(results))
    #print("=============")
    #print(results[0])
    generated_texts = list(map(lambda x: x.outputs[0].text, results))
    #print(generated_texts)
    generated_token_ids = list(map(lambda x: x.outputs[0].token_ids, results))
    #print(generated_token_ids)
    log_probs: List[SampleLogprobs] = list(map(lambda x: x.outputs[0].logprobs, results))
    generated_log_probs = list(map(lambda x: list(map(
        lambda y: y[0][y[1]].logprob, zip(x[0], x[1])
    )), zip(log_probs, generated_token_ids)))
    destroy_vllm_instance(vllm_model)
    del vllm_model
    del generate_params
    gc.collect()
    return generated_texts, generated_token_ids, generated_log_probs


def make_vllm_instance(llm: str, input_args):
    from vllm import LLM
    #print(2)
    model_from_args = input_args.pop('model', None)
    #print(22)
    model = llm if model_from_args is None else model_from_args
    #print(222)
    init_params = inspect.signature(LLM.__init__).parameters.values()
    #print(2222)
    keyword_init_params = [param.name for param in init_params if param.kind == param.KEYWORD_ONLY]
    #print(22222)
    input_kwargs = {}
    for param in keyword_init_params:
        v = input_args.pop(param, None)
        if v is not None:
            input_kwargs[param] = v
    if 'trust_remote_code' not in input_kwargs:
        input_kwargs['trust_remote_code'] = True
    #print(222222)
    #print(model)
    #print(input_kwargs)
    #print(**input_kwargs)
    return LLM(model=model,trust_remote_code=True)


def destroy_vllm_instance(vllm_instance):
    if torch.cuda.is_available():
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
        )

        destroy_model_parallel()
        del vllm_instance
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    else: 
        del vllm_instance
        gc.collect()
