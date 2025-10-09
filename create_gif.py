from moviepy import VideoFileClip

# Load video file
clip = VideoFileClip("nao_nao_exo.mp4")

# Optionally, trim the clip (e.g., first 5 seconds)
# clip = clip.subclip(0, 5)

# Resize if needed (e.g., reduce size by 50%)
clip = clip.resized(0.5)

# Write to GIF
clip.write_gif("images/nao_exo.gif", fps=120)

#ffmpeg -i human_play.mp4 -vf "fps=30,scale=600:-1:flags=lanczos" -gifflags -transdiff -y human_play.gif