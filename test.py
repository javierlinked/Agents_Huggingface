from tools import youtube_analyze

# Test the youtube_analyze search directly
result = youtube_analyze("In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?", "https://www.youtube.com/watch?v=L1vXCYZAYYM")
print("youtube search result:")
print(result)