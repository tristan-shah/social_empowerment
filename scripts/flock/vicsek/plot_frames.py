import subprocess


ratio = 200 / 80 ## physical / video

physical_time = 200
video_time = physical_time * (1/ratio)

print(video_time)

# subprocess.run([
#     "ffmpeg",
#     "-i", "vid.mp4",
#     "-ss", str(video_time),
#     "-frames:v", "1",
#     f"frame_{physical_time}s.jpg"
# ], check = True)


subprocess.run([
    "ffmpeg",
    "-i", "vid_sharp.mp4",
    "-ss", str(video_time),
    "-frames:v", "1",
    "-vf", "scale=iw*2:ih*2",   # 2x upscale — adjust multiplier as needed
    "-q:v", "1",                 # best JPEG quality (1=best, 31=worst)
    f"frame_{physical_time}s.jpg"
], check=True)