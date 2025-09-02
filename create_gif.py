from moviepy import VideoFileClip

# Load video file
clip = VideoFileClip("videos/robot_play.mp4")

# Optionally, trim the clip (e.g., first 5 seconds)
# clip = clip.subclip(0, 5)

# Resize if needed (e.g., reduce size by 50%)
clip = clip.resized(0.5)

# Write to GIF
clip.write_gif("images/robot_play.gif", fps=120)
