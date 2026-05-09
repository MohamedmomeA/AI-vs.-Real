import gradio as gr

from utils.predict import predict

title = "AI vs Real Image Detector"

description = """
Upload an image and the model will predict whether it is AI-generated or real.

Model: ConvNeXt Tiny
Validation Accuracy: 94.79%
"""

ai_examples = [
    ["examples_2/AI_animals_1.png"],
    ["examples_2/AI_animals_2.png"],
    ["examples_2/AI_city_1.png"],
    ["examples_2/AI_city_2.png"],
    # ["examples_2/AI_people_1.png"],
    # ["examples_2/AI_people_2.png"],
    ["examples_2/AI_nature_1.png"],
    ["examples_2/AI_nature_2.png"],
    ["examples_2/AI_food_1.png"],
    ["examples_2/AI_food_2.png"],
    # ["examples_2/AI_1.jpg"],
    ["examples_2/AI_2.jpg"],
    ["examples_2/AI_3.jpg"],
    ["examples_2/AI_4.jpg"],
    ["examples_2/AI_5.jpg"],
    ["examples_2/AI_8.png"],
    ["examples_2/AI_9.jpg"],

    ["examples_2/AI_15.png"],
    ["examples_2/AI_16.jpg"],
    ["examples_2/AI_19.png"],
    ["examples_2/AI_23.png"],
    ["examples_2/AI_24.jpg"],
    ["examples_2/AI_26.jpg"],
    ["examples_2/AI_27.png"],
    ["examples_2/AI_31.jpg"],
    ["examples_2/AI_32.jpg"],

    ["examples_2/AI_35.jpg"],
    ["examples_2/AI_36.png"],
    ["examples_2/AI_38.jpg"],
    ["examples_2/AI_39.png"],
    ["examples_2/AI_100.png"],

]

real_examples = [
    ["examples_2/real_animals_1.jpg"],
    ["examples_2/real_animals_2.jpg"],
    ["examples_2/real_city_1.jpg"],
    ["examples_2/real_city_2.jpg"],
    ["examples_2/real_people_1.jpg"],
    # ["examples_2/real_people_2.jpg"],
    ["examples_2/real_nature_1.jpg"],
    ["examples_2/real_nature_2.jpg"],
    ["examples_2/real_food_1.jpg"],
    ["examples_2/real_1.jpg"],
    ["examples_2/real_2.jpg"],
    # ["examples_2/real_4.jpg"],
    ["examples_2/real_6.jpg"],
    ["examples_2/real_7.jpg"],
    ["examples_2/real_8.jpg"],
    ["examples_2/real_9.jpg"],

    ["examples_2/real_11.jpg"],
    ["examples_2/real_13.jpg"],
    ["examples_2/real_14.jpg"],
    ["examples_2/real_15.jpg"],
    ["examples_2/real_17.jpg"],
    ["examples_2/real_18.jpg"],
    ["examples_2/real_19.jpg"],

    ["examples_2/real_20.jpg"],
    ["examples_2/real_21.jpg"],
    ["examples_2/real_23.jpg"],
    ["examples_2/real_26.jpg"],
    ["examples_2/real_31.jpg"],
    ["examples_2/real_33.jpg"],
    ["examples_2/real_34.jpg"],

    ["examples_2/real_35.jpg"],
    ["examples_2/real_36.jpg"],
    ["examples_2/real_39.jpg"],
    ["examples_2/real_40.jpg"],
    ["examples_2/real_100.jpg"],
    
]

with gr.Blocks() as demo:

    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    image_input = gr.Image(type="pil", label="Upload Image")

    output = gr.Label(num_top_classes=2, label="Prediction")

    predict_btn = gr.Button("Predict")

    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=output
    )

    with gr.Tab("AI-Generated Examples"):
        gr.Examples(
            examples=ai_examples,
            inputs=image_input,
            outputs=output,
            fn=predict,
            cache_examples=False
        )

    with gr.Tab("Real Image Examples"):
        gr.Examples(
            examples=real_examples,
            inputs=image_input,
            outputs=output,
            fn=predict,
            cache_examples=False
        )

# demo.launch(share=True) # uncomment If you want to share it out of the local pc and comment the "demo.launch(server_name="0.0.0.0")" line

demo.launch(server_name="0.0.0.0")