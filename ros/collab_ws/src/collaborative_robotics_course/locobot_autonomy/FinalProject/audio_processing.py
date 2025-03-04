#!/usr/bin/env python3

# ROS2 Import
import rclpy                    # ROS2 client library
from rclpy.node import Node     # ROS2 node baseclass
import time
import os
from std_msgs.msg import String, Bool

# API import
import sounddevice as sd
import wave
from google.cloud import speech_v1p1beta1 as speech
import google.generativeai as genai

# robot path
json_key_path = "/home/locobot/Naixiang/ME326FinalProject/ros/collab_ws/src/collaborative_robotics_course/locobot_autonomy/FinalProject/APIKeys/NaixiangKey.json"
# simulation path
# json_key_path = "/home/ubuntu/Desktop/ME326/ros/collab_ws/src/collaborative_robotics_course/locobot_autonomy/FinalProject/APIKeys/NaixiangKey.json"
# test_audio_path = "/home/ubuntu/Desktop/ME326/ros/collab_ws/src/collaborative_robotics_course/locobot_autonomy/FinalProject/recorded_audio3.wav"

print("Current json key path:", json_key_path)
if os.path.exists(json_key_path):
    print("File exsits!")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path
else:
    print("Not found! Exiting...")
    exit(1)

class AudioProcess(Node):
    def __init__(self) -> None:
        # give it a default node name
        super().__init__("AudioProcess")
        self.get_logger().info("AudioProcess node has been started!")
        self.filename = "recorded_audio.wav"
        # self.filename = test_audio_path
        self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        self.prompt = ("In the given voice transcript, identify what the object is that the user wants."
                       "Return only the object in lowercase and do not include any whitespaces, punctuation, or new."
                       "If all contents of voice transcript has some fruit unrelated words, just responce no objects."
                       "If the voice transcript has the actions, please also return the action in the same pattern of objects in this format: objects, action"
                       "Here is the voice transcript: ")
        
        self.item = ""

        self.audio_publisher = self.create_publisher(String, 'AudioItem', 10)

        self.modify_audio_subscriber = self.create_subscription(Bool, 'ModifyAudioStatus', self.modify_audio_callback, 10)

        self.timer_publish_audio = self.create_timer(1.0, self.audio_item_publisher)
        self.modify_audio_pub = self.create_publisher(Bool, 'ModifyAudioStatus', 10)

        self.modify_flag = 0

        # self.timer_receive_audio = self.create_timer(1.0, self.audio_input_check_callback)

    def audio_item_publisher(self):
        msg = String()
        msg.data = self.item
        self.audio_publisher.publish(msg)
    
    def modify_audio_callback(self, msg):
        self.get_logger().info(f"Received ModifyAudioStatus: {msg.data}")
        # edge detector
        if msg.data == True and self.modify_flag == 0:
            self.modify_flag = 1
            self.audio_process()
            self.get_logger().info("Finish Recording")

            mdodify_audio_msg = Bool()
            mdodify_audio_msg.data = False
            self.modify_audio_pub.publish(mdodify_audio_msg)
        else:
            self.modify_flag = 0

    def audio_process(self):
        # 1. Record from microphone
        self.record_audio(self.filename, duration=5, sample_rate=16000)
        # 2. Transcribe the recorded audio
        transcription = self.transcribe_audio(self.filename)
        # 3. process the transcription
        raw_item = self.generate_content(transcription)
        # print("Gemini raw output:", repr(raw_item)) 
        self.item = raw_item.strip().lower().rstrip(".!?,")
    
    def record_audio(self, filename, duration=5, sample_rate=16000):
        """
        Records audio from the microphone for a given duration (in seconds) and
        sample rate, then saves it to a WAV file.
        """
        print("Recording started...")
        # Record audio (mono channel)
        recorded_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='int16'  # 16-bit PCM
        )
        sd.wait()  # Wait until recording is complete
        print("Recording complete. Saving audio...")

        # Write the audio data to a WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)                 # mono
            wf.setsampwidth(2)                # 2 bytes (16 bits)
            wf.setframerate(sample_rate)
            wf.writeframes(recorded_data.tobytes())
        print(f"Audio saved to {filename}")

    def transcribe_audio(self, filename, sample_rate=16000):
        """
        Uses Google Cloud Speech-to-Text to transcribe the given WAV audio file.
        Returns the transcription as a string.
        """
        client = speech.SpeechClient()

        # Read the audio file into memory
        with open(filename, 'rb') as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            sample_rate_hertz=sample_rate,
            language_code='en-US',
            enable_automatic_punctuation=True,
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        )

        response = client.recognize(config=config, audio=audio)

        # Extract the transcription from the response
        full_transcript = []
        for result in response.results:
            # Each "result" may contain multiple alternatives
            full_transcript.append(result.alternatives[0].transcript)

        return " ".join(full_transcript)
    
    def generate_content(self, text):
        print("Generating content for:", text)
        input = self.prompt + text
        response = self.gemini_model.generate_content(input)
        return response.text

if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = AudioProcess()           # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    node.destroy_node()     # clean up node before shutdown
    rclpy.shutdown()        # clean up after node exits
