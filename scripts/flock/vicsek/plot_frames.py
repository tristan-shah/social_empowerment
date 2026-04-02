import subprocess


ratio = 200 / 80 ## physical / video

physical_time = 75
video_time = physical_time * (1/ratio)

print(video_time)

subprocess.run([
    "ffmpeg",
    "-i", "vid.mp4",
    "-ss", str(video_time),
    "-frames:v", "1",
    f"frame_{physical_time}s.jpg"
], check = True)