import os
from openai import OpenAI
import base64
from PIL import Image
from dotenv import load_dotenv
from io import BytesIO
from articulate_anything.agent.critic.joint_prediction.joint_critic import get_frames_from_video
import argparse
import json

IN_CONTEXT_EXAMPLE = """
```json
{
    "part_1": {"coordinates": "x1=25.2, y1=74.5",
    "reasoning": "Top drawer that can slide out."},
    "part_2": {"coordinates": "x2"="25.4", "y2"="27.3",
    "reasoning": "Middle drawer that can slide out."},
    "part_3": {"coordinates": "x3"="25.4", "y3"="46.3",
    "reasoning": "Bottom drawer that can slide out."},
    "part_4": {"coordinates": "x4"="25.4", "y4"="61.3",
    "reasoning": "Left front foot."},
    "part_5": {"coordinates": "x5"="59.6", y5="80.2",
    "reasoning": "Right front foot."}
}
```
"""

def make_prompt(input_path):
    content = [
            {
                "type": "text",
                "text": "Here is the image of size (800,800) of the object: "
            },
        ]
    
    content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_to_base64(input_path)}"
                }
            },
        )

    return content

def make_example(example_img):
    content = [
            {
                "type": "text",
                "text": "To help you accomplish this task I will provide here an example of an input image and one plausible correct response. The image has size (1080,1920):"
            },
        ]
    
    content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_to_base64(example_img)}"
                }
            },
        )
    
    content.append(
            {
                "type": "text",
                "text": "This is an acceptable response: " + IN_CONTEXT_EXAMPLE
            },
        )

    return content

def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def identify_movable_parts(img):
    # Initialize OpenAI client
    load_dotenv()
    client = OpenAI(api_key=os.environ.get("API_KEY"), base_url="https://ai-gateway.mytkhgroup.com/")
    
    # Encode image
    # base64_image = encode_image_to_base64(image_path)

    system_instruction =  """
    You will be presented with an image of an object. Your task is to identify all movable or articulated parts in this image. Focus on components that can rotate, slide, or be manipulated.
    Point out the moving parts as well as the static base component of the object. Give as response the image coordinates of the center of each of these parts based on the resolution of the image.

    Provide your response in the following format, it MUST be provided in json format like this:
    ```json
    {
        "part_0":{"coordinates": "x0=40.0, y0=12.7",
        "reasoning": "Explanation of why this part was selected and how the articulation can be inferred"},

        "part_1":{"coordinates": "x1=244.2, y1=545.1",
        "reasoning": "Explanation of why this part was selected and how the articulation can be inferred"},
        ...
    }
    ```

    """
    content = make_prompt(Image.open(img))

    example_img = "/home/link/DreMa/third_party/articulate-anything/datasets/in-the-wild-dataset/images/drawer_RL_Bench.jpeg"

    example = make_example(Image.open(example_img))
    
    # Prepare the message
    messages = [
        {
            "role": "system", "content": system_instruction},
            {"role": "user",
             "content": example
             },
            {
            "role": "user",
            "content": content
        }
    ]
    
    # Call the API
    response = client.chat.completions.create(
        model="claude-3-5-sonnet-latest",
        messages=messages,
        temperature=0.5
    )
    
    return response.choices[0].message.content

def str_to_json(api_response):
    import re
    import json
    """Extracts the json found between triple backticks (```) in a string."""
    # pattern = r"```json\n([\s\S]*?)```"
    pattern = r"```json\n(.*?)```"
    matches = re.findall(pattern, api_response, re.DOTALL)


    if matches:
        # Return the first match of the Python code block, removing whitespace
        # return matches[0].strip()
        return json.loads(matches[0])
    else:
        return None  # Indicate no code found

def format_result(json_response):
    result = []
    # print(json_response.values())
    for part in list(json_response.values()):#[::2]:
        coords = part['coordinates']
        x, y = coords.split(', ')
        x = x.split('=')[1]
        y = y.split('=')[1]
        result.append([x,y])
    return result
    # for part in json_response:
    #     '. '.join([result, list(part.values())[0]])

def detect_articulated_parts(video_path):
    # video_path = "/home/link/DreMa/third_party/articulate-anything/datasets/in-the-wild-dataset/videos/drawer_RL_Bench.mp4"

    response = identify_movable_parts(video_path)
    # print(response)

    response_json = str_to_json(response)
    formatted_result = format_result(response_json)
    # print("Movable parts identified:")
    # print(formatted_result)
    return formatted_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('object', default='laptop_real')
    args = parser.parse_args()
    input_path = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{args.object}"

    rgb_images = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) 
                        if f.startswith(f"render_{args.object}") and f.endswith(".png")])
    
    results = []
    for i, img in enumerate(rgb_images):

        response = detect_articulated_parts(img)
        print(f"Result for img {i}: {response}")
        results.append(response)

    with open(os.path.join(input_path, 'articulated_points.json'), 'w') as f:
        json.dump(results, f)
    print(f"Points saved")

    
