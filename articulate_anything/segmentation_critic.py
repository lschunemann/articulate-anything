import os
import re
import json
import argparse
from typing import List, Tuple, Optional, Dict, Any

from PIL import Image

from articulate_anything.utils.prompt_utils import setup_vlm_model


def load_results(results_dir: str) -> List[dict]:
    results = []
    if not os.path.isdir(results_dir):
        return results
    for name in sorted(os.listdir(results_dir)):
        if name.startswith("render_") and name.endswith("_results.json"):
            path = os.path.join(results_dir, name)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                results.append(data)
            except Exception:
                continue
    return results


def load_images(input_dir: str, masks_dir: str) -> Tuple[List[Image.Image], List[Image.Image]]:
    raw_images: List[Image.Image] = []
    vis_images: List[Image.Image] = []
    if os.path.isdir(input_dir):
        for name in sorted(os.listdir(input_dir)):
            if name.startswith("render_") and name.endswith(".png"):
                try:
                    raw_images.append(Image.open(os.path.join(input_dir, name)).convert("RGB"))
                except Exception:
                    pass
    if os.path.isdir(masks_dir):
        for name in sorted(os.listdir(masks_dir)):
            if name.startswith("render_") and name.endswith("_vis.jpg"):
                try:
                    vis_images.append(Image.open(os.path.join(masks_dir, name)).convert("RGB"))
                except Exception:
                    pass
    return raw_images, vis_images


def find_video_file(input_dir: str) -> Optional[str]:
    if not os.path.isdir(input_dir):
        return None
    exts = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    for name in sorted(os.listdir(input_dir)):
        lower = name.lower()
        if any(lower.endswith(ext) for ext in exts):
            return os.path.join(input_dir, name)
    return None


def sample_video_frames(video_path: str, num_frames: int = 5) -> List[Image.Image]:
    print(f"DEBUG: Attempting to load video from {video_path}")
    if not os.path.exists(video_path):
        print(f"ERROR: Video file does not exist: {video_path}")
        return []
    
    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV not available for video processing")
        return []
    
    frames: List[Image.Image] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return frames
    
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"DEBUG: Video has {length} frames")
    if length <= 0:
        print("ERROR: Video has no frames")
        cap.release()
        return frames
    
    # Evenly spaced indices
    indices = [max(0, min(length - 1, int(i * (length - 1) / max(1, num_frames - 1)))) for i in range(num_frames)]
    print(f"DEBUG: Sampling frames at indices: {indices}")
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"WARNING: Could not read frame {idx}")
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        frames.append(pil_img)
    
    cap.release()
    print(f"DEBUG: Successfully loaded {len(frames)} video frames")
    return frames


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    # Look for the first top-level JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    snippet = match.group(0)
    try:
        return json.loads(snippet)
    except Exception:
        return None


def run_vlm_judgment(
    model_name: str,
    api_key: Optional[str],
    raw_images: List[Image.Image],
    vis_images: List[Image.Image],
    video_frames: List[Image.Image],
    prev_prompt: str,
    temperature: float = 0.2,
    max_images: int = 8,
) -> Optional[Dict[str, Any]]:
    print(f"DEBUG: Starting VLM judgment with model={model_name}")
    print(f"DEBUG: API key present: {api_key is not None}")
    print(f"DEBUG: Raw images: {len(raw_images)}, Vis images: {len(vis_images)}, Video frames: {len(video_frames)}")
    print(f"DEBUG: Previous prompt: '{prev_prompt}'")
    
    system_instruction = (
        "You are a vision-language critic for segmentation quality. "
        "Given original rendered views, their corresponding segmentation overlays, and a few frames from the input video, "
        "assess how well the masks isolate the intended parts described by the prompt. "
        "Return a strict JSON object with fields: "
        "score (float 0.0-1.0, higher is better), "
        "refined_prompt (string with concise, improved segmentation targets separated by periods)."
    )

    client = setup_vlm_model(model_name=model_name, system_instruction=system_instruction, api_key=api_key)
    if client is None:
        print("ERROR: Failed to setup VLM model - check API key and model name")
        return None

    # Prepare prompt parts: brief text + images (interleave raw and vis)
    prompt_parts: List[Any] = []
    prompt_parts.append(
        (
            "Previous segmentation targets: " + prev_prompt + "\n" +
            "Evaluate segmentation quality from 0 to 1 and propose a refined prompt. "
            "Only output JSON."
        )
    )

    limit = min(max_images, len(raw_images), len(vis_images))
    print(f"DEBUG: Processing {limit} image pairs")
    for i in range(limit):
        prompt_parts.append(f"Original view #{i+1}")
        prompt_parts.append(raw_images[i])
        prompt_parts.append(f"Segmentation overlay view #{i+1}")
        prompt_parts.append(vis_images[i])

    # Append sampled video frames
    if video_frames:
        print(f"DEBUG: Adding {len(video_frames)} video frames")
        prompt_parts.append("Sampled frames from the input video (for correctness of target part):")
        for i, vf in enumerate(video_frames[:5]):
            prompt_parts.append(f"Video frame #{i+1}")
            prompt_parts.append(vf)
    else:
        print("DEBUG: No video frames to add")

    try:
        print("DEBUG: Calling VLM...")
        response = client.generate_content(prompt_parts, generation_config={"temperature": temperature})
        print(f"DEBUG: VLM response: {response.text[:200]}...")
        parsed = extract_json(response.text)
        print(f"DEBUG: Parsed JSON: {parsed}")
        return parsed
    except Exception as e:
        print(f"ERROR: VLM call failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("object", type=str)
    parser.add_argument("--min_score", type=float, default=0.55)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--vlm_model", type=str, default="gemini-1.5-flash")
    args = parser.parse_args()

    seg_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/{args.object}"
    input_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{args.object}"
    prompt_file = os.path.join(input_dir, "segmentation_targets.txt")

    results = load_results(seg_dir)

    # Load assets for VLM
    prev_prompt = ""
    if os.path.isfile(prompt_file):
        try:
            with open(prompt_file, "r") as f:
                prev_prompt = f.read().strip()
        except Exception:
            prev_prompt = ""
    raw_images, vis_images = load_images(input_dir, seg_dir)
    # video_path = find_video_file(input_dir)
    video_path = f"/home/link/DreMa/third_party/articulate-anything/datasets/RLBench/videos/{''.join(args.object.split('_')[:-1])+'_RLBench'}.mp4"
    video_frames: List[Image.Image] = sample_video_frames(video_path, num_frames=5) if video_path else []

    vlm_score: Optional[float] = None
    vlm_refined: Optional[str] = None
    api_key = os.environ.get("API_KEY")
    vlm_result = None
    
    print(f"DEBUG: Checking prerequisites - raw_images: {len(raw_images)}, vis_images: {len(vis_images)}, prev_prompt: '{prev_prompt}', video_frames: {len(video_frames)}")
    
    if raw_images and vis_images and prev_prompt:
        vlm_result = run_vlm_judgment(
            model_name=args.vlm_model,
            api_key=api_key,
            raw_images=raw_images,
            vis_images=vis_images,
            video_frames=video_frames,
            prev_prompt=prev_prompt,
        )
    else:
        print("ERROR: Missing required inputs for VLM - need raw_images, vis_images, and prev_prompt")
    
    if isinstance(vlm_result, dict):
        vlm_score = vlm_result.get("score")
        rp = vlm_result.get("refined_prompt")
        if isinstance(vlm_score, (int, float)):
            vlm_score = max(0.0, min(1.0, float(vlm_score)))
        if isinstance(rp, str) and len(rp.strip()) > 0:
            vlm_refined = rp.strip()
        print(f"DEBUG: VLM returned score={vlm_score}, refined_prompt='{vlm_refined}'")
    else:
        print("ERROR: VLM did not return valid result")

    # Final score is strictly from the VLM
    final_score = vlm_score if isinstance(vlm_score, (int, float)) else 0.0
    print(json.dumps({
        "score": final_score,
        "vlm_score": vlm_score
    }, indent=2))

    if final_score >= args.min_score:
        exit(0)

    # Prepare for retry: refined prompt priority -> VLM refined, else keep previous
    can_retry = args.attempt < args.max_retries
    if can_retry:
        new_prompt = None
        if vlm_refined and len(vlm_refined) > 0:
            new_prompt = vlm_refined
        elif prev_prompt:
            new_prompt = prev_prompt

        if new_prompt:
            try:
                # Backup and overwrite
                with open(prompt_file + ".bak", "w") as f:
                    f.write(prev_prompt)
            except Exception:
                pass
            with open(prompt_file, "w") as f:
                f.write(new_prompt)
            print(f"Refined prompt: {new_prompt}")
            exit(2)

    exit(1)


if __name__ == "__main__":
    main()


