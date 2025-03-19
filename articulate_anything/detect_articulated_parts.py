import os
from openai import OpenAI
import base64
from PIL import Image
from dotenv import load_dotenv
from io import BytesIO
from articulate_anything.agent.critic.joint_prediction.joint_critic import get_frames_from_video

def make_prompt(input_path):
    video = get_frames_from_video(
        input_path,
        num_frames=5,
        video_encoding_strategy="individual",
        # width=self.cfg.simulator.camera_params.width,
        # height=self.cfg.simulator.camera_params.height,
    )
    content = [
            {
                "type": "text",
                "text": "Here is the video of the object: "
            },
        ]
    
    for image in video:
        content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image_to_base64(image)}"
                    }
                },
        )

    return content

def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def identify_movable_parts(video_path):
    # Initialize OpenAI client
    load_dotenv()
    client = OpenAI(api_key=os.environ.get("API_KEY"), base_url="https://ai-gateway.mytkhgroup.com/")
    
    # Encode image
    # base64_image = encode_image_to_base64(image_path)

    #system_instruction =  """
    #You will be presented with a video of an object. Your task is to identify and list all movable or articulated parts in this video. Focus on components that can rotate, slide, or be manipulated.

    #Provide your response in the following format:
    #'''json
    #{
    #    {"part_0": "<description>", # eg. "The top one out of three wooden drawers that slides in and out."
    #    "reasoning": "Explanation of why this part was selected and how the articulation can be inferred"},

    #    {"part_1": "<description>", # eg. "A lid on top of the object that can be twisted to be undone."
    #    "reasoning": "Explanation of why this part was selected and how the articulation can be inferred"},
    #    ...
    #}
    #'''
    #"""
    system_instruction = """
    You have a good understanding of the structure of articulated objects. Your job is to assist the user to analyze the structure of an object. Specifically, the user will give you a video of an articulated object, and your task is to recognize the main parts of that object. You should give your answer in the following format:
    ```part_list
    (1) part_name: name of the part; description: a brief description about the part, and how it moves
    (2) part_name: name of the part; description: a brief description about the part, and how it moves
    ...
    ```
    Remember:
    (1) Do not answer anything not asked.
    (2) Your answer should be purely based on the input video, do not imagine anything.
    (3) If there are multiple parts with the same semantic, just add one part to the list. For example, if there are four wheels, just add one part whose name is wheel.
    
    """

    content = make_prompt(video_path)
    
    # Prepare the message
    messages = [
        {
            "role": "system", "content": system_instruction},
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
    pattern = r"```part_list\n([\s\S]*?)```"
    matches = re.findall(pattern, api_response, re.DOTALL)


    if matches:
        # Return the first match of the Python code block, removing whitespace
        return matches[0].strip()
        #return json.loads(matches[0])
        # return 
    else:
        return api_response
        # return None  # Indicate no code found

def format_result(json_response):
    result = ''
    for part in list(json_response.values())[::2]:
        result = '. '.join([part, result])
    result = '. '.join([list(json_response.values())[-1], result])
    return result
    # for part in json_response:
    #     '. '.join([result, list(part.values())[0]])

def format_result_str(str_response):
    result = ''

    print(str_response)
    print(str_response.split('\n'))

    if str_response.startswith('Based'):
        for part_whole in str_response.strip().split('\n')[1:]:
            if part_whole == '': continue
            print(part_whole)
            part_, desc = part_whole.split(';')
            part = part_.split(':')[1]
            description = desc.split(':')[1]
            result = '. '.join([part, result])
    else:
        for part_whole in str_response.split('\n'):
            print(part_whole)
            part_, desc = part_whole.split(';')
            part = part_.split(':')[0]
            description = desc.split(':')[1]
            result = '. '.join([part, result])
    return result

def detect_articulated_parts(video_path):
    # video_path = "/home/link/DreMa/third_party/articulate-anything/datasets/in-the-wild-dataset/videos/drawer_RL_Bench.mp4"

    response = identify_movable_parts(video_path)
    # response = """
    # ```json
    # {
    #     "part_0": "Top drawer with gray handle that slides in and out horizontally",
    #     "reasoning": "The video shows the top drawer can be pulled outward along horizontal rails, as evidenced by the sliding motion and handle placement",

    #     "part_1": "Middle drawer with gray handle that slides in and out horizontally", 
    #     "reasoning": "Like the top drawer, the middle drawer is shown sliding out horizontally with a handle for pulling",

    #     "part_2": "Bottom drawer with gray handle that slides in and out horizontally",
    #     "reasoning": "The bottom drawer matches the functionality of the other drawers, with a handle and horizontal sliding motion along rails"
    # }
    # ```"""
    response_json = str_to_json(response)
    #formatted_result = format_result(response_json)
    formatted_result = format_result_str(response_json)
    # print("Movable parts identified:")
    # print(formatted_result)
    return formatted_result

if __name__ == '__main__':
    # video_path = "/home/link/DreMa/third_party/articulate-anything/datasets/in-the-wild-dataset/videos/drawer_RL_Bench.mp4"
    video_path = "/home/link/DreMa/third_party/articulate-anything/datasets/in-the-wild-dataset/videos/microwave_RL_Bench.mp4"

    response = detect_articulated_parts(video_path)
    print(response)