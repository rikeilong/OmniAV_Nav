from typing import List, Optional
import inspect

import os

import torch
import uvicorn
from fastapi import Body, FastAPI
from pydantic import BaseModel
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from qwen_omni_utils import process_mm_info
from rao_nav_inference.latent_navigation_reasoning import LNRConfig, select_lnr_frames
from rao_nav_inference.paths import QWEN_MODEL_DIR

app = FastAPI()

model_dir = str(QWEN_MODEL_DIR)
USE_LNR = os.environ.get("QWEN_USE_LNR", "0").strip().lower() in {"1", "true", "yes", "on"}
LNR_LATENT_TOKENS = int(os.environ.get("QWEN_LNR_LATENT_TOKENS", "8"))
LNR_OPTIMIZATION_STEPS = int(os.environ.get("QWEN_LNR_OPTIMIZATION_STEPS", "4"))
LNR_STEP_SIZE = float(os.environ.get("QWEN_LNR_STEP_SIZE", "0.1"))
LNR_TOP_K = int(os.environ.get("QWEN_LNR_TOP_K", "4"))
LNR_MAX_HISTORY = int(os.environ.get("QWEN_LNR_MAX_HISTORY", "20"))


def _resolve_device_map() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    
    return os.environ.get("QWEN_DEVICE", "cuda:0")


def _load_model():
    processor_local = Qwen2_5OmniProcessor.from_pretrained(model_dir)
    device_map = _resolve_device_map()
    dtype = torch.bfloat16 if device_map.startswith("cuda") else torch.float32
    kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
    }

    try:
        model_local = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_dir,
            **kwargs,
        )
    except Exception as e:
        print(f"[WARN] 加载失败，尝试更保守配置: {e}")
        try:
            
            model_local = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=dtype,
                device_map=device_map,
            )
        except Exception as e2:
            print(f"[WARN] CUDA eager 仍失败，回退 CPU 模式: {e2}")
            model_local = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.float32,
                device_map="cpu",
                attn_implementation="eager",
            )
    return processor_local, model_local


processor, model = _load_model()


class InferenceRequest(BaseModel):
    temperature: float = 0.9
    question: str = ""
    user_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    video_path: str
    audio_path: Optional[str] = None
    multi_video_path: Optional[str] = "None"
    history_actions: List[str] = ["None"]
    collision: bool = False
    max_new_tokens: int = 256
    use_lnr: Optional[bool] = None
    lnr_goal: Optional[str] = None
    history_image_paths: List[str] = []


def _run_inference(payload: InferenceRequest):
    global processor, model
    processor_sig = inspect.signature(processor.__call__)
    processor_params = set(processor_sig.parameters.keys())
    
    
    supports_audio = "audio" in processor_params or "audios" in processor_params

    default_system_prompt = (
        "You are an audio-visual navigation agent in a 3D indoor environment. "
        "A target object is emitting a sound. Based on the target audio, the current RGB observation, "
        "and the navigation history, predict the next action. "
        "You must output exactly one action from: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP. "
        "Do not provide any explanation."
    )
    sp = payload.system_prompt
    if sp is not None and str(sp).strip():
        system_prompt = str(sp).strip()
    else:
        system_prompt = default_system_prompt

    multi_video_path = payload.multi_video_path or "None"
    has_audio = payload.audio_path is not None and payload.audio_path != ""
    use_audio = has_audio and supports_audio
    history_text = ", ".join(payload.history_actions[-3:]) if payload.history_actions else "None"
    base_question = (payload.user_prompt or payload.question or "").strip()
    if not base_question:
        base_question = (
            "The target object is emitting the input audio. "
            f"Recent action history: {history_text}. "
            "Based on the target audio and the recent visual observations, what is the next action?"
        )
    collision_suffix = ""
    if payload.collision:
        collision_suffix = (
            " Previous action collided with an obstacle. "
            "To avoid repeated collision, prefer TURN_LEFT or TURN_RIGHT before MOVE_FORWARD."
        )

    lnr_enabled = USE_LNR if payload.use_lnr is None else bool(payload.use_lnr)
    lnr_result = {"selected_paths": [], "selected_indices": [], "rewards": []}
    lnr_error = ""
    if lnr_enabled and payload.history_image_paths:
        try:
            lnr_result = select_lnr_frames(
                model=model,
                processor=processor,
                history_image_paths=payload.history_image_paths,
                predicted_audio_goal=payload.lnr_goal or base_question,
                config=LNRConfig(
                    latent_tokens=LNR_LATENT_TOKENS,
                    optimization_steps=LNR_OPTIMIZATION_STEPS,
                    step_size=LNR_STEP_SIZE,
                    top_k=LNR_TOP_K,
                    max_history=LNR_MAX_HISTORY,
                ),
            )
        except Exception as error:
            lnr_error = str(error)
            print(f"[WARN] LNR failed and current-frame inference will be used: {error}")
    if lnr_enabled:
        print(
            f"[LNR] history={len(payload.history_image_paths)} "
            f"selected={lnr_result['selected_indices']} rewards={lnr_result['rewards']}"
        )

    def _user_prompt(image_tag_count: int) -> str:
        return f"<audio>{'<image>' * image_tag_count}{base_question}{collision_suffix}"

    if "None" not in multi_video_path:
        prompt_text = _user_prompt(image_tag_count=1)
        user_content = [
            {"type": "text", "text": prompt_text},
            {"type": "image", "image": payload.video_path},
            {"type": "video", "video": multi_video_path},
        ]
        if use_audio:
            user_content.append({"type": "audio", "audio": payload.audio_path})
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]
    else:
        selected_images = list(lnr_result["selected_paths"])
        image_paths = selected_images + [payload.video_path]
        prompt_text = _user_prompt(image_tag_count=len(image_paths))
        user_content = [{"type": "text", "text": prompt_text}]
        user_content.extend({"type": "image", "image": path} for path in image_paths)
        if use_audio:
            user_content.append({"type": "audio", "audio": payload.audio_path})
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

    processor_inputs = {
        "text": text,
        "return_tensors": "pt",
        "padding": True,
    }
    
    if "images" in processor_params and images is not None:
        processor_inputs["images"] = images
    if "videos" in processor_params and videos is not None:
        processor_inputs["videos"] = videos
    if "audio" in processor_params and audios is not None:
        processor_inputs["audio"] = audios
    elif "audios" in processor_params and audios is not None:
        processor_inputs["audios"] = audios
    elif has_audio:
        print("[WARN] 当前 processor 不支持 audio/audios 参数，音频不会进入 processor。")

    inputs = processor(**processor_inputs)
    inputs = inputs.to(model.device).to(model.dtype)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max(int(payload.max_new_tokens), 1),
        eos_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
        return_dict_in_generate=True,
        temperature=max(float(payload.temperature), 1e-5),
    )
    trimmed_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs.sequences)]
    response = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    response = response.strip()
    print(response)
    return {
        "response": response,
        "lnr_enabled": lnr_enabled,
        "lnr_selected_indices": lnr_result["selected_indices"],
        "lnr_rewards": lnr_result["rewards"],
        "lnr_error": lnr_error,
    }


@app.post("/v1/omni/inference")
async def omni_inference(payload: InferenceRequest):
    return _run_inference(payload)


@app.get("/health")
async def health():
    return {"status": "ok", "model_dir": model_dir, "lnr_enabled": USE_LNR}



@app.get("/v1/omni/inference")
async def omni_inference_legacy(
    temperature: float = Body(0.9, alias="temperature"),
    question: str = Body("", alias="question"),
    video_path: str = Body(..., alias="video_path"),
    audio_path: Optional[str] = Body(None, alias="audio_path"),
    multi_video_path: str = Body("None", alias="multi_video_path"),
    history_actions: List[str] = Body(["None"], alias="history_actions"),
    collision: bool = Body(False, alias="collision"),
    user_prompt: Optional[str] = Body(None, alias="user_prompt"),
    system_prompt: Optional[str] = Body(None, alias="system_prompt"),
    max_new_tokens: int = Body(256, alias="max_new_tokens"),
    use_lnr: Optional[bool] = Body(None, alias="use_lnr"),
    lnr_goal: Optional[str] = Body(None, alias="lnr_goal"),
    history_image_paths: List[str] = Body([], alias="history_image_paths"),
):
    payload = InferenceRequest(
        temperature=temperature,
        question=question,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        video_path=video_path,
        audio_path=audio_path,
        multi_video_path=multi_video_path,
        history_actions=history_actions,
        collision=collision,
        max_new_tokens=max_new_tokens,
        use_lnr=use_lnr,
        lnr_goal=lnr_goal,
        history_image_paths=history_image_paths,
    )
    return _run_inference(payload)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("QWEN_HOST", "127.0.0.1"),
        port=int(os.environ.get("QWEN_PORT", "6006")),
        workers=1,
    )
