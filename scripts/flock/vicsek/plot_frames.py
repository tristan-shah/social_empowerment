import subprocess


ratio = 80 / 200

times = [1, 10, 20, 30, 40, 45]


for time in times:

    video_time = time * (1/ratio)

    subprocess.run([
        "ffmpeg",
        "-ss", str(video_time),
        "-i", "vid.mp4",
        "-frames:v", "1",
        f"frame_{str(time)}s.jpg"
    ], check = True)