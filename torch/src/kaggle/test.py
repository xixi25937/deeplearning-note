import kagglehub

# Download latest version
path = kagglehub.competition_download('california-house-prices')

print("Path to competition files:", path)