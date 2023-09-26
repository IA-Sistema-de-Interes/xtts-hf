import gradio as gr
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1")
tts.to("cuda")


def predict(prompt, language, audio_file_pth, agree):
    if agree == True:
        tts.tts_to_file(
            text=prompt,
            file_path="output.wav",
            speaker_wav=audio_file_pth,
            language=language_mapping.get(language, "Desconocido"),
        )

        return (
            gr.make_waveform(
                audio="output.wav",
            ),
            "output.wav",
        )
    else:
        gr.Warning("Please accept the Terms & Condition!")


title = """Coqui🐸 XTTS <a href="https://www.youtube.com/channel/UC1ejkTHsiq8aQAeYIZyIyeg">Visita mi Canal: IA (Sistema de interes)</a>"""

description = """
<a href="https://huggingface.co/coqui/XTTS-v1">XTTS</a> es un modelo de generación de voz que te permite clonar voces en diferentes idiomas usando solo un clip de audio rápido de 3 segundos. 
<br/>
XTTS se basa en investigaciones anteriores, como Tortoise, con innovaciones arquitectónicas adicionales y capacitación para hacer posible la clonación de voz en varios idiomas y la generación de voz multilingüe.
<br/>
Este es el mismo modelo que impulsa nuestra aplicación creadora <a href="https://coqui.ai">Coqui Studio</a>, así como <a href="https://docs.coqui.ai"> API Coqui</a>. En producción aplicamos modificaciones para hacer posible la transmisión de baja latencia.
<br/>
Deje una estrella en Github <a href="https://github.com/coqui-ai/TTS">🐸TTS</a>, donde reside nuestro código de entrenamiento e inferencia de código abierto.
<br/>
<p>Para una inferencia más rápida sin esperar en la cola, debe duplicar este espacio y actualizar a GPU a través de la configuración.
<br/>
<a href="https://huggingface.co/spaces/coqui/xtts?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
</p>
"""

article = """
<div style='margin:20px auto;'>
<p>Al utilizar esta demostración, acepta los términos de la Licencia de modelo público de Coqui en https://coqui.ai/cpml</p>
</div>
"""

language_mapping = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Polish': 'pl',
    'Turkish': 'tr',
    'Russian': 'ru',
    'Dutch': 'nl',
    'Czech': 'cs',
    'Arabic': 'ar',
    'Chinese (Simplified)': 'zh'
}

gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(
            label="Text Prompt",
            info="One or two sentences at a time is better",
            value="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
        ),
        gr.Dropdown(
            label="Language",
            info="Select an output language for the synthesised speech",
            choices=[
                "Arabic",
                "Chinese (Simplified)",
                "Czech",
                "Dutch",
                "English",
                "French",
                "German",
                "Italian",
                "Polish",
                "Portuguese",
                "Russian",
                "Spanish",
                "Turkish",
            ],
            max_choices=1,
            value="en",
        ),
        gr.Audio(
            label="Reference Audio",
            info="Click on the ✎ button to upload your own target speaker audio",
            type="filepath",
            value="examples/female.wav",
        ),
        gr.Checkbox(
            label="Agree",
            value=False,
            info="I agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml",
        ),
    ],
    outputs=[
        gr.Video(label="Waveform Visual"),
        gr.Audio(label="Synthesised Audio"),
    ],
    title=title,
    description=description,
    article=article,
    examples="",
).queue().launch(share=True)
