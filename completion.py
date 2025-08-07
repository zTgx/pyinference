
from fastapi.responses import StreamingResponse
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

engine_args = AsyncEngineArgs(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    tensor_parallel_size=2,  # 双GPU并行
    max_model_len=8192,
    gpu_memory_utilization=0.9,
    quantization="gptq",     # 4-bit量化
    enforce_eager=True       # 减少显存碎片[2](@ref)
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

async def do_completion(request):
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=0.9
    )

    # 流式输出
    if request.stream:
        async def stream_results():
            async for output in engine.generate(
                request.prompt, 
                sampling_params,
                request_id=f"req_{hash(request.prompt)}"
            ):
                yield output.outputs[0].text

        return StreamingResponse(stream_results())

    # 非流式输出
    output = await engine.generate(
        request.prompt,
        sampling_params,
        request_id=f"req_{hash(request.prompt)}"
    )