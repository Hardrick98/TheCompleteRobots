from moviepy import VideoFileClip

# Load video file
clip = VideoFileClip("high_five.mp4")

# Optionally, trim the clip (e.g., first 5 seconds)
# clip = clip.subclip(0, 5)

# Resize if needed (e.g., reduce size by 50%)
# clip = clip.resize(0.5)

# Write to GIF
clip.write_gif("high_five.gif", fps=60)
